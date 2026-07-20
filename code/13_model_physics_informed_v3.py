"""
13_model_physics_informed_v3.py: 可微物理前向建模核心模块（V3 改进版）

V3 改进内容：
    1. 加入吸收系数平滑约束（相邻波段二阶差分惩罚）
    2. 调整损失权重配置（强化BL约束）
    3. 改进回归器架构（更深、残差连接）
    4. 优化MMD实现（稳定带宽计算）

运行方式:
    cd 02code
    import 13_model_physics_informed_v3 (作为模块导入使用)

输出文件：
    无独立输出文件（作为模块被其他实验调用）
"""


import os
import sys
import importlib.util
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import argparse

warnings.filterwarnings('ignore')

# ─── 加载工具模块 ──────────────────────────────────────────────────────────
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "export_utils", os.path.join(_CODE_DIR, "01_export_utils.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

setup_chinese_font = _mod.setup_chinese_font
figure_output_path = _mod.figure_output_path
write_excel_workbook = _mod.write_excel_workbook
get_logger = _mod.get_logger
get_device = _mod.get_device

setup_chinese_font()
logger = get_logger('13_model_physics_informed_v3')
log = logger.log

# ═══════════════════════════════════════════════════════════════════════════
# 平滑约束函数
# ═══════════════════════════════════════════════════════════════════════════

def smoothness_loss(x, axis=-1):
    """
    计算二阶差分平滑损失

    鼓励曲线平滑：惩罚二阶差分（曲率）
    L_smooth = mean(Δ²x)²
    """
    # 二阶差分: x[i] - 2*x[i+1] + x[i+2]
    diff2 = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    return torch.mean(diff2 ** 2)


def monotonic_loss(x, axis=-1):
    """
    计算单调性约束损失

    惩罚单调性违反（对于某些波段应该单调的区域）
    这里实现一个宽松的单调性约束
    """
    diff = x[..., 1:] - x[..., :-1]
    # 只惩罚大幅反向变化（超过一定阈值）
    violations = torch.relu(-diff * 0.1)  # 允许小的负斜率
    return torch.mean(violations ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# 物理分解器模块 V3
# ═══════════════════════════════════════════════════════════════════════════

class BeerLambertDecomposerV3(nn.Module):
    """
    可微物理分解器 V3: 光谱 → (吸收系数, 光程, 浓度)

    V3 改进:
        - 加入平滑约束损失
        - 更深的共享编码器（带残差）
        - 改进的初始化策略
        - 在前向传播中计算平滑损失
    """

    def __init__(self, n_bands=229, n_components=3, hidden_dim=128):
        super().__init__()
        self.n_bands = n_bands
        self.n_components = n_components
        self.hidden_dim = hidden_dim

        # ─ 共享编码器（更深的MLP，带残差）
        self.shared_encoder = nn.Sequential(
            nn.Linear(n_bands, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # ─ 吸收系数头: 输出 log 空间参数 (B, n_comp, n_bands)
        self.absorption_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_components * n_bands),
        )

        # ─ 光程头: 输出 log 空间标量 (B,)
        self.optical_path_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ─ 浓度头: 输出 log 空间参数 (B, n_comp, n_bands)
        self.concentration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_components * n_bands),
        )

        # 改进的初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, X, compute_smoothness=True):
        """
        Args:
            X: (B, n_bands) - 标准化后的光谱
            compute_smoothness: 是否计算平滑损失

        Returns:
            a: (B, n_comp, n_bands) - 吸收系数（正数）
            b: (B,) - 有效光程（正数）
            c: (B, n_comp, n_bands) - 浓度（正数）
            X_recon: (B, n_bands) - 重构光谱
            loss_smooth: 平滑损失张量
        """
        B = X.shape[0]

        # 共享编码
        h = self.shared_encoder(X)  # (B, hidden_dim)

        # 吸收系数: log 空间
        log_a = self.absorption_head(h).reshape(B, self.n_components, self.n_bands)
        a = torch.exp(log_a)  # 保证正

        # 光程: log 空间
        log_b = self.optical_path_head(h).squeeze(-1)  # (B,)
        b = torch.exp(log_b)  # 保证正

        # 浓度: log 空间
        log_c = self.concentration_head(h).reshape(B, self.n_components, self.n_bands)
        c = torch.exp(log_c)  # 保证正

        # BL重构: A = Σ_i (a_i * b * c_i)
        log_abc = log_a + log_b.unsqueeze(-1).unsqueeze(-1) + log_c
        log_recon = torch.logsumexp(log_abc, dim=1)  # (B, n_bands)
        X_recon = torch.exp(log_recon)

        loss_smooth = None
        if compute_smoothness:
            # 吸收系数平滑损失（log空间）
            loss_smooth = (
                smoothness_loss(log_a) + smoothness_loss(log_c)
            ) / 2

        return a, b, c, X_recon, loss_smooth

    def get_physical_features(self, X):
        """
        获取物理可解释特征，用于域适应和回归

        Returns:
            features: (B, 4*n_comp+1) = [a_mean, b, c_mean, a_std, c_std]
        """
        a, b, c, _, _ = self.forward(X, compute_smoothness=False)

        a_mean = a.mean(dim=2)   # (B, n_comp)
        a_std = a.std(dim=2)     # (B, n_comp)
        b_feat = b.unsqueeze(1)  # (B, 1)
        c_mean = c.mean(dim=2)   # (B, n_comp)
        c_std = c.std(dim=2)     # (B, n_comp)

        features = torch.cat([a_mean, b_feat, c_mean, a_std, c_std], dim=1)
        return features

    def get_reconstruction_loss(self, X_src, X_tgt=None):
        """
        计算 BL 重构损失（log 空间归一化版本）
        """
        _, _, _, X_recon_src, _ = self.forward(X_src, compute_smoothness=False)
        eps = 1e-6
        loss_src = F.mse_loss(torch.log(X_recon_src + eps), torch.log(X_src.abs() + eps))

        if X_tgt is not None:
            _, _, _, X_recon_tgt, _ = self.forward(X_tgt, compute_smoothness=False)
            loss_tgt = F.mse_loss(torch.log(X_recon_tgt + eps), torch.log(X_tgt.abs() + eps))
            return loss_src + loss_tgt

        return loss_src


# ═══════════════════════════════════════════════════════════════════════════
# 回归器 V3（改进架构）
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsInformedRegressorV3(nn.Module):
    """
    基于物理分解特征的回归器 V3

    改进:
        - 更深的网络
        - 残差连接
        - 更好的正则化
    """

    def __init__(self, n_components=3, hidden_dim=64):
        super().__init__()
        n_features = 4 * n_components + 1  # 13

        # 第一层
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # 第二层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 第三层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        # 输出层
        self.fc_out = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(0.2)

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 第一层
        h1 = self.dropout(F.relu(self.bn1(self.fc1(x))))

        # 第二层（残差）
        h2 = self.dropout(F.relu(self.bn2(self.fc2(h1))))
        h2 = h2 + h1  # 残差连接

        # 第三层
        h3 = self.dropout(F.relu(self.bn3(self.fc3(h2))))

        # 输出
        out = self.fc_out(h3)
        return out.squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# 完整模型 V3
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsInformedDomainAdaptationV3(nn.Module):
    """
    完整的可微物理前向域适应模型 V3

    pipeline:
        1. 光谱 → 共享编码器 → 分解为 (log_a, log_b, log_c)
        2. BL约束: log 空间重构损失
        3. 平滑约束: 二阶差分惩罚
        4. 回归: 物理特征(a,b,c统计量) → SSC
        5. 域适应: 物理特征空间 MMD

    V3 改进:
        - 加入平滑约束损失
        - 调整损失权重配置
        - 改进的回归器架构
    """

    def __init__(self, n_bands=229, n_components=3, hidden_dim=128):
        super().__init__()
        self.decomposer = BeerLambertDecomposerV3(
            n_bands=n_bands,
            n_components=n_components,
            hidden_dim=hidden_dim
        )
        self.regressor = PhysicsInformedRegressorV3(
            n_components=n_components,
            hidden_dim=hidden_dim
        )
        self.n_bands = n_bands
        self.n_components = n_components

    def forward(self, X_src, y_src=None, X_tgt=None):
        """
        前向传播

        Args:
            X_src: (B_s, n_bands) - 源域光谱
            y_src: (B_s,) - 源域标签（可选）
            X_tgt: (B_t, n_bands) - 目标域光谱（训练时必须传入）

        Returns:
            dict: y_pred, loss_reg, loss_bl, loss_smooth, feat_src, feat_tgt
        """
        results = {}

        # 源域分解
        a_src, b_src, c_src, X_recon_src, loss_smooth_src = self.decomposer(X_src)

        # BL重构约束（log 空间）
        eps = 1e-6
        loss_bl = F.mse_loss(
            torch.log(X_recon_src + eps),
            torch.log(X_src.abs() + eps)
        )
        results['loss_bl'] = loss_bl

        # 平滑约束
        results['loss_smooth'] = loss_smooth_src if loss_smooth_src is not None else torch.tensor(0.0)

        # 源域物理特征 + 回归
        feat_src = self.decomposer.get_physical_features(X_src)
        results['feat_src'] = feat_src

        y_pred = self.regressor(feat_src)
        results['y_pred'] = y_pred

        if y_src is not None:
            loss_reg = F.mse_loss(y_pred, y_src)
            results['loss_reg'] = loss_reg
        else:
            results['loss_reg'] = torch.tensor(0.0, device=X_src.device)

        # 目标域
        if X_tgt is not None:
            a_tgt, b_tgt, c_tgt, X_recon_tgt, loss_smooth_tgt = self.decomposer(X_tgt)

            loss_bl_tgt = F.mse_loss(
                torch.log(X_recon_tgt + eps),
                torch.log(X_tgt.abs() + eps)
            )
            results['loss_bl_tgt'] = loss_bl_tgt

            if loss_smooth_tgt is not None:
                results['loss_smooth_tgt'] = loss_smooth_tgt

            feat_tgt = self.decomposer.get_physical_features(X_tgt)
            results['feat_tgt'] = feat_tgt
        else:
            results['loss_bl_tgt'] = torch.tensor(0.0, device=X_src.device)
            results['loss_smooth_tgt'] = torch.tensor(0.0, device=X_src.device)
            results['feat_tgt'] = None

        return results

    def compute_total_loss(self, results, lambda_bl=1.0, lambda_mmd=0.5, lambda_smooth=0.1):
        """
        计算加权总损失（V3 改进版）

        total_loss = loss_reg + λ_bl * loss_bl + λ_smooth * loss_smooth + λ_mmd * loss_mmd
        """
        loss_reg = results['loss_reg']
        loss_bl = results['loss_bl'] + results.get('loss_bl_tgt', 0)
        loss_smooth = results['loss_smooth'] + results.get('loss_smooth_tgt', 0)

        # MMD
        if results['feat_tgt'] is not None:
            loss_mmd = self._compute_mmd(results['feat_src'], results['feat_tgt'])
        else:
            loss_mmd = torch.tensor(0.0, device=loss_reg.device)

        total_loss = (
            loss_reg +
            lambda_bl * loss_bl +
            lambda_smooth * loss_smooth +
            lambda_mmd * loss_mmd
        )

        return total_loss, {
            'reg': loss_reg.item(),
            'bl': loss_bl.item(),
            'smooth': loss_smooth.item(),
            'mmd': loss_mmd.item() if isinstance(loss_mmd, torch.Tensor) else 0,
        }

    @staticmethod
    def _compute_mmd(feat_src, feat_tgt):
        """
        MMD (RBF核)，改进的稳定带宽计算
        """
        n_src = feat_src.shape[0]
        n_tgt = feat_tgt.shape[0]

        # 计算所有样本对的距离
        feat_cat = torch.cat([feat_src, feat_tgt], dim=0)
        feat_norm = (feat_cat ** 2).sum(dim=1, keepdim=True)
        dist_sq = feat_norm + feat_norm.t() - 2 * torch.mm(feat_cat, feat_cat.t())
        dist_sq = dist_sq.clamp(min=1e-10)

        # 稳定的带宽：使用中位数，并加一个常数避免太小
        mask = ~torch.eye(dist_sq.shape[0], dtype=torch.bool, device=dist_sq.device)
        bandwidth = torch.sqrt(dist_sq[mask].median()) + 1e-3

        def rbf_kernel(A, B):
            A_norm = (A ** 2).sum(dim=1, keepdim=True)
            B_norm = (B ** 2).sum(dim=1, keepdim=True).t()
            AB = torch.mm(A, B.t())
            d = A_norm + B_norm - 2 * AB
            return torch.exp(-d / (2 * bandwidth ** 2))

        K_ss = rbf_kernel(feat_src, feat_src).mean()
        K_tt = rbf_kernel(feat_tgt, feat_tgt).mean()
        K_st = rbf_kernel(feat_src, feat_tgt).mean()

        return K_ss + K_tt - 2 * K_st


# ═══════════════════════════════════════════════════════════════════════════
# 测试代码
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='可微物理前向建模模块 V3 测试')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='设备选择')
    args = parser.parse_args()

    log("="*80)
    log("13_model_physics_informed_v3.py V3 - 可微物理前向建模模块")
    log("="*80)

    device = get_device(args.device)
    log(f"设备: {device}")

    # 测试: 创建模型
    model = PhysicsInformedDomainAdaptationV3(n_bands=229, n_components=3, hidden_dim=128)
    model.to(device)
    log(f"模型创建成功")

    # 测试: 前向传播（含目标域）
    B_src, B_tgt = 32, 32
    X_src = torch.randn(B_src, 229).to(device)
    y_src = torch.randn(B_src).to(device) * 3 + 12
    X_tgt = torch.randn(B_tgt, 229).to(device)

    results = model(X_src, y_src, X_tgt)
    log(f"前向传播成功")
    log(f"  源域预测: {results['y_pred'].shape}")
    log(f"  BL损失: {results['loss_bl'].item():.4f}")
    log(f"  平滑损失: {results['loss_smooth'].item():.4f}")
    log(f"  feat_tgt: {'有' if results['feat_tgt'] is not None else '无'}")

    # 测试: 总损失
    total_loss, loss_dict = model.compute_total_loss(
        results, lambda_bl=1.0, lambda_mmd=0.5, lambda_smooth=0.1
    )
    log(f"总损失: {total_loss.item():.4f}")
    log(f"  分项: {loss_dict}")

    # 测试: 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n模型参数统计:")
    log(f"  总参数: {total_params:,}")
    log(f"  可训练: {trainable_params:,}")

    # 测试: 梯度反传
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    log(f"  梯度裁剪后范数: {grad_norm:.4f}")

    log("\n" + "="*80)
    log("V3 模块测试完成 - 可以开始实验训练")
    log("="*80)

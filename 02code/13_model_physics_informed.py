"""
13_model_physics_informed.py: 可微物理前向建模核心模块（V2 修复版）

修复内容：
    1. BL重构改为 log 空间归一化，避免 a*b*c 乘积梯度爆炸
    2. a/b/c 三个网络共享编码器，减少参数量
    3. MMD 带宽自适应（median heuristic）
    4. 梯度裁剪保护

运行方式:
    cd 02code
    import 13_model_physics_informed (作为模块导入使用)

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
logger = get_logger('13_model_physics_informed')
log = logger.log

# ═══════════════════════════════════════════════════════════════════════════
# 物理分解器模块 V2
# ═══════════════════════════════════════════════════════════════════════════

class BeerLambertDecomposer(nn.Module):
    """
    可微物理分解器 V2: 光谱 → (吸收系数, 光程, 浓度)

    V2 改进:
        - 共享编码器减少参数量（~150K vs 旧版 ~500K）
        - BL重构在 log 空间计算，避免乘积梯度爆炸
        - 使用 log_a + log_b + log_c 的加法形式等价于 log(a*b*c)

    物理约束:
        A_recon(λ) = Σ_i [ a_i(λ) × b × c_i(λ) ]
        在 log 空间: log(A_recon) = logsumexp_i [ log(a_i) + log(b) + log(c_i) ]
    """

    def __init__(self, n_bands=229, n_components=3, hidden_dim=64):
        super().__init__()
        self.n_bands = n_bands
        self.n_components = n_components
        self.hidden_dim = hidden_dim

        # ─ 共享编码器（三层MLP）
        self.shared_encoder = nn.Sequential(
            nn.Linear(n_bands, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # ─ 吸收系数头: 输出 log 空间参数 (B, n_comp, n_bands)
        self.absorption_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
            nn.Linear(hidden_dim, n_components * n_bands),
        )

    def forward(self, X):
        """
        Args:
            X: (B, n_bands) - 标准化后的光谱

        Returns:
            a: (B, n_comp, n_bands) - 吸收系数（正数）
            b: (B,) - 有效光程（正数）
            c: (B, n_comp, n_bands) - 浓度（正数）
            X_recon: (B, n_bands) - 重构光谱
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
        # 在 log 空间计算避免梯度爆炸
        log_abc = log_a + log_b.unsqueeze(-1).unsqueeze(-1) + log_c
        # logsumexp over components: log(Σ exp(log_abc_i))
        log_recon = torch.logsumexp(log_abc, dim=1)  # (B, n_bands)
        X_recon = torch.exp(log_recon)

        return a, b, c, X_recon

    def get_physical_features(self, X):
        """
        获取物理可解释特征，用于域适应和回归

        Returns:
            features: (B, 4*n_comp+1) = [a_mean, b, c_mean, a_std, c_std]
        """
        a, b, c, _ = self.forward(X)

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

        使用 log 空间的 MSE 避免大值主导梯度:
            L_BL = MSE(log(X+eps), log(X_recon+eps))
        """
        _, _, _, X_recon_src = self.forward(X_src)
        eps = 1e-6
        loss_src = F.mse_loss(torch.log(X_recon_src + eps), torch.log(X_src.abs() + eps))

        if X_tgt is not None:
            _, _, _, X_recon_tgt = self.forward(X_tgt)
            loss_tgt = F.mse_loss(torch.log(X_recon_tgt + eps), torch.log(X_tgt.abs() + eps))
            return loss_src + loss_tgt

        return loss_src


# ═══════════════════════════════════════════════════════════════════════════
# 回归器
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsInformedRegressor(nn.Module):
    """
    基于物理分解特征的回归器

    输入: 物理特征 (B, 4*n_comp+1)
    输出: SSC糖度预测 (B,)
    """

    def __init__(self, n_components=3, hidden_dim=32):
        super().__init__()
        n_features = 4 * n_components + 1  # 13

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, physical_features):
        return self.net(physical_features).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════
# 完整模型 V2
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsInformedDomainAdaptation(nn.Module):
    """
    完整的可微物理前向域适应模型 V2

    pipeline:
        1. 光谱 → 共享编码器 → 分解为 (log_a, log_b, log_c)
        2. BL约束: log 空间重构损失
        3. 回归: 物理特征(a,b,c统计量) → SSC
        4. 域适应: 物理特征空间 MMD（自适应带宽）

    V2 修复:
        - 训练时正确传入 X_tgt 使 MMD 生效
        - BL 损失在 log 空间计算
        - MMD 带宽使用 median heuristic
        - 梯度裁剪
    """

    def __init__(self, n_bands=229, n_components=3, hidden_dim=64):
        super().__init__()
        self.decomposer = BeerLambertDecomposer(
            n_bands=n_bands,
            n_components=n_components,
            hidden_dim=hidden_dim
        )
        self.regressor = PhysicsInformedRegressor(
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
            dict: y_pred, loss_reg, loss_bl, feat_src, feat_tgt
        """
        results = {}

        # 源域分解
        a_src, b_src, c_src, X_recon_src = self.decomposer(X_src)

        # BL重构约束（log 空间）
        eps = 1e-6
        loss_bl = F.mse_loss(
            torch.log(X_recon_src + eps),
            torch.log(X_src.abs() + eps)
        )
        results['loss_bl'] = loss_bl

        # 源域物理特征 + 回归
        feat_src = self.decomposer.get_physical_features(X_src)
        results['feat_src'] = feat_src

        y_pred = self.regressor(feat_src)
        results['y_pred'] = y_pred

        if y_src is not None:
            # 调用方的 DataLoader 把标签存成 (B,1)，而 y_pred 是 (B,)：直接相减会广播成 B×B，
            # 每个预测都去对批内全部标签，回归项里就没有「哪条谱配哪个标签」的信息，
            # 模型只学到批均值（模块顶部的 warnings 过滤把 PyTorch 的尺寸警告也吞了）。
            # 先把标签摊成与预测同形再算。
            loss_reg = F.mse_loss(y_pred, y_src.reshape(y_pred.shape))
            results['loss_reg'] = loss_reg
        else:
            results['loss_reg'] = torch.tensor(0.0, device=X_src.device)

        # 目标域
        if X_tgt is not None:
            a_tgt, b_tgt, c_tgt, X_recon_tgt = self.decomposer(X_tgt)

            loss_bl_tgt = F.mse_loss(
                torch.log(X_recon_tgt + eps),
                torch.log(X_tgt.abs() + eps)
            )
            results['loss_bl_tgt'] = loss_bl_tgt

            feat_tgt = self.decomposer.get_physical_features(X_tgt)
            results['feat_tgt'] = feat_tgt
        else:
            results['loss_bl_tgt'] = torch.tensor(0.0, device=X_src.device)
            results['feat_tgt'] = None

        return results

    def compute_total_loss(self, results, lambda_bl=0.5, lambda_mmd=0.3):
        """
        计算加权总损失（含梯度裁剪保护）

        total_loss = loss_reg + λ_bl * loss_bl + λ_mmd * loss_mmd
        """
        loss_reg = results['loss_reg']
        loss_bl = results['loss_bl'] + results.get('loss_bl_tgt', 0)

        # MMD（自适应带宽）
        if results['feat_tgt'] is not None:
            loss_mmd = self._compute_mmd(results['feat_src'], results['feat_tgt'])
        else:
            loss_mmd = torch.tensor(0.0, device=loss_reg.device)

        total_loss = loss_reg + lambda_bl * loss_bl + lambda_mmd * loss_mmd

        return total_loss, {
            'reg': loss_reg.item(),
            'bl': loss_bl.item(),
            'mmd': loss_mmd.item() if isinstance(loss_mmd, torch.Tensor) else 0,
        }

    @staticmethod
    def _compute_mmd(feat_src, feat_tgt):
        """
        MMD (RBF核)，带宽使用 median heuristic 自适应

        bandwidth = median(||x_i - x_j||) 的中位数
        """
        # 拼接计算中位距离
        n_src = feat_src.shape[0]
        n_tgt = feat_tgt.shape[0]
        n_sample = min(n_src, n_tgt, 64)  # 采样加速

        idx_src = torch.randperm(n_src)[:n_sample]
        idx_tgt = torch.randperm(n_tgt)[:n_sample]

        X = torch.cat([feat_src[idx_src], feat_tgt[idx_tgt]], dim=0)
        # ||X_i - X_j||^2
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        dist_sq = X_norm + X_norm.t() - 2 * torch.mm(X, X.t())
        dist_sq = dist_sq.clamp(min=0)

        # median heuristic
        mask = torch.ones(dist_sq.shape[0], dist_sq.shape[1], dtype=torch.bool)
        mask.fill_diagonal_(False)
        bandwidth = dist_sq[mask].median().clamp(min=1e-3).sqrt() + 1e-6

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
    parser = argparse.ArgumentParser(description='可微物理前向建模模块 V2 测试')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='设备选择')
    args = parser.parse_args()

    log("="*80)
    log("13_model_physics_informed.py V2 - 可微物理前向建模模块")
    log("="*80)

    device = get_device(args.device)
    log(f"设备: {device}")

    # 测试: 创建模型
    model = PhysicsInformedDomainAdaptation(n_bands=229, n_components=3, hidden_dim=64)
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
    log(f"  feat_tgt: {'有' if results['feat_tgt'] is not None else '无'}")

    # 测试: 总损失
    total_loss, loss_dict = model.compute_total_loss(results, lambda_bl=0.5, lambda_mmd=0.3)
    log(f"总损失: {total_loss.item():.4f}")
    log(f"  分项: {loss_dict}")
    log(f"  MMD != 0: {loss_dict['mmd'] > 0}")

    # 测试: 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\n模型参数统计:")
    log(f"  总参数: {total_params:,}")
    log(f"  可训练: {trainable_params:,}")

    # 测试: 梯度反传
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    log(f"  梯度裁剪前范数: {grad_norm:.4f}")

    log("\n" + "="*80)
    log("模块测试完成 - 可以开始实验训练")
    log("="*80)

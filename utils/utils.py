import torch
from torch import nn
from torch.nn import Module, Embedding, Linear, Dropout, MaxPool1d, Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


device = "cpu" if not torch.cuda.is_available() else "cuda"

class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size),
                # Dropout(self.dropout),
            )
    def forward(self, in_fea):
        return self.FFN(in_fea)

def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)

def lt_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)

def pos_encode(seq_len):
    """ position Encoding
    """
    return torch.arange(seq_len).unsqueeze(0).to(device)

def get_clones(module, N):
    """ Cloning nn modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class InvariantFeatureLearner(nn.Module):
    def __init__(self, dim, alpha_stability=0.3, max_window=16):
        super().__init__()
        self.dim = dim
        self.alpha_stability = alpha_stability
        self.max_window = max_window

        self.hidden_dim = dim
        self.invariant_dim = dim * 2 
        
        # [Fix 1] 序列编码器输入维度修正
        # Input: Mean(Seq)[d] + Mean(Op)[d] = 2d
        self.sequence_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim * 2), # Modified from dim * 4
            nn.LayerNorm(dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim)
        )
        
        # 频域特征提取器
        self.spectral_processor = self._build_spectral_processor(dim)
        
        # 特征稳定性约束模块
        self.stability_constraint = self._build_stability_module(dim)

        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.hidden_dim // 2, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.invariant_dim)
        )
        
        self.alpha_stability = alpha_stability
        
    def _build_spectral_processor(self, dim):
        """构建频域特征处理器"""
        # [Fix 2] 频域输入维度修正
        # Joint (2d) -> 5 features (LowMag, LowPhase, StdMag, StdPhase, Entropy) -> 5 * 2d = 10d
        return nn.Sequential(
            nn.Linear(dim * 10, dim), # Modified from dim * 20
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim // 2)
        )
    
    def _build_stability_module(self, dim):
        """构建特征稳定性约束模块"""
        return nn.Sequential(
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 4, dim // 8)
        )

    def sequence_to_spectral(self, sequence_emb, operation_emb, skill_emb=None):
        """将序列数据转换为频域表示"""
        if skill_emb is not None:
            joint_representation = torch.cat([sequence_emb, operation_emb, skill_emb], dim=-1)
        else:
            joint_representation = torch.cat([sequence_emb, operation_emb], dim=-1)
        B, L, D_joint = joint_representation.shape
        W = self.max_window
        if joint_representation.abs().sum() < 1e-10:
            return torch.zeros(B, L, D_joint * 5, device=joint_representation.device)        
        joint_padded = F.pad(joint_representation, (0, 0, W - 1, 0)) # [B, L + W - 1, D_joint]
        feature_count = 5 # LowMag, LowPhase, StdMag, StdPhase, Entropy
        input_channels = joint_representation.size(-1) # 2 * dim
        expected_dim = input_channels * feature_count # 10 * dim
        

        windows = joint_padded.unfold(1, W, 1)
        joint_fft = torch.fft.fft(windows, dim=-1)
        magnitude = torch.abs(joint_fft)  

        eps = 1e-7
        magnitude_safe = magnitude + eps
            
            # Phase Unwrapping Logic
        angle = torch.angle(joint_fft)
            
        # 提取低频分量
        low_freq_idx = max(1, W // 4)
       # [B, L, D_joint] <- 对窗口维(dim=-1)的前部分取平均
        low_freq_mag = magnitude[..., :low_freq_idx].mean(dim=-1)
        
        # 相位平均 (使用复数求和再转回角度，避免相位卷绕问题)
        # exp(1j * angle) -> sum -> angle
        low_freq_complex = torch.complex(
            torch.cos(angle[..., :low_freq_idx]), 
            torch.sin(angle[..., :low_freq_idx])
        )
        low_freq_phase = torch.angle(low_freq_complex.mean(dim=-1))

        # 5.2 波动性特征 (Std Dev)
        # 在窗口维度上计算标准差
        mag_std = magnitude.std(dim=-1)
        
        # 环形数据的标准差计算较复杂，这里简化为 cos/sin 的标准差组合
        cos_std = torch.cos(angle).std(dim=-1)
        sin_std = torch.sin(angle).std(dim=-1)
        phase_std = torch.sqrt(cos_std**2 + sin_std**2)

        # 5.3 谱熵 (Spectral Entropy)
        # 输入 magnitude 最后一个维度是频率分布
        mag_entropy = self._calculate_spectral_entropy(magnitude)

        # 6. 组合所有特征
        # 形状都是 [B, L, D_joint]
        # 拼接后: [B, L, 5 * D_joint]
        spectral_features = torch.cat([
            low_freq_mag,
            low_freq_phase,
            mag_std,
            phase_std,
            mag_entropy
        ], dim=-1)
        
        
        return spectral_features
    
    def _calculate_spectral_entropy(self, magnitude):
            # magnitude shape: [B, L, D_joint, W]
            eps = 1e-10
            p = magnitude / (magnitude.sum(dim=-1, keepdim=True) + eps)
            entropy = -torch.sum(p * torch.log(p + eps), dim=-1) # [B, L, D_joint]
            return entropy
    
    def extract_invariant_features(self, sequence_emb, operation_emb, skill_emb=None):
        B, L, D = sequence_emb.shape
        joint_temp = torch.cat([sequence_emb, operation_emb], dim=-1)
        counts = torch.arange(1, L + 1, device=sequence_emb.device).view(1, L, 1)
        causal_mean = torch.cumsum(joint_temp, dim=1) / counts # [B, L, 2D]
        raw_features = self.sequence_encoder(causal_mean) # [B, L, D]
        

        # [B, L, 10D]
        spectral_input = self.sequence_to_spectral(sequence_emb, operation_emb, skill_emb)
        #[B, L, D//2]
        spectral_features_all = self.spectral_processor(spectral_input)
        

        stability_scores = self.stability_constraint(spectral_features_all)
        stability_weights = torch.sigmoid(stability_scores.mean(dim=-1, keepdim=True))
        weighted_spectral_features = spectral_features_all * stability_weights
        
        # [B, L, 1.5D] 
        combined_features = torch.cat([raw_features, weighted_spectral_features], dim=-1)
        
        # [B, L, 2D]
        invariant_features = self.feature_fusion(combined_features)
        

        if L > 1:

            step_sim = F.cosine_similarity(invariant_features[:, 1:], invariant_features[:, :-1], dim=-1)
            stability_loss = -torch.mean(step_sim) * self.alpha_stability
        else:
            stability_loss = torch.tensor(0.0, device=device)
        
        return invariant_features, stability_loss
    
    def forward(self, sequence_emb, operation_emb, skill_emb=None, mode='train'):
        invariant_features, stability_loss = self.extract_invariant_features(
            sequence_emb, operation_emb, skill_emb
        )

        return invariant_features, stability_loss
    



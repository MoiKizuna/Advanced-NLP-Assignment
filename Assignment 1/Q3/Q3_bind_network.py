#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3: Bind Network 实现
基于论文《ImageBind-LLM: Multi-modality Instruction Tuning》中的图3
实现图像特征到转换后图像特征的映射网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS normalization
        norm_x = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (norm_x + self.eps))


class BindNetwork(nn.Module):
    """
    Bind Network 实现
    
    网络结构：
    1. 初始线性变换 (W0)
    2. 三个重复的核心块 (x3)，每个包含：
       - 归一化 (Norm)
       - 并行路径：W2 和 (W1 + SiLU)
       - 元素级乘法 (⊗)
       - 线性变换 (W3)
       - 残差连接 (⊕)
    
    输入: image_feature (图像特征)
    输出: transformed_image_feature (转换后的图像特征)
    """
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 4096, 
                 num_blocks: int = 3, use_rms_norm: bool = True):
        """
        初始化 Bind Network
        
        Args:
            input_dim: 输入特征维度 (默认 1024，对应 ImageBind 输出)
            hidden_dim: 隐藏层维度 (默认 4096，对应 LLaMA 维度)
            num_blocks: 重复块的数量 (默认 3)
            use_rms_norm: 是否使用 RMSNorm (默认 True)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        
        # 初始线性变换 W0
        self.W0 = nn.Linear(input_dim, hidden_dim)
        
        # 定义归一化层类型
        if use_rms_norm:
            norm_layer = RMSNorm
        else:
            norm_layer = nn.LayerNorm
        
        # 三个重复的核心块
        self.norms = nn.ModuleList([
            norm_layer(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.W1_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 4) for _ in range(num_blocks)
        ])
        
        self.W2_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 4) for _ in range(num_blocks)
        ])
        
        self.W3_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 4, hidden_dim) for _ in range(num_blocks)
        ])
        
        # SiLU 激活函数
        self.silu = nn.SiLU()
    
    def forward(self, image_feature):
        """
        前向传播
        
        Args:
            image_feature: 输入图像特征 [batch_size, seq_len, input_dim]
            
        Returns:
            transformed_image_feature: 转换后的图像特征 [batch_size, seq_len, hidden_dim]
        """
        # 初始线性变换 W0
        x = self.W0(image_feature)
        
        # 三个重复的核心块
        for i in range(self.num_blocks):
            # 保存输入用于残差连接
            residual = x
            
            # 归一化
            x_norm = self.norms[i](x)
            
            # 并行路径
            # 路径1: W2
            w2_out = self.W2_layers[i](x_norm)
            
            # 路径2: W1 + SiLU
            w1_out = self.W1_layers[i](x_norm)
            silu_out = self.silu(w1_out)
            
            # 元素级乘法 (⊗)
            gate_out = w2_out * silu_out
            
            # 线性变换 W3
            x = self.W3_layers[i](gate_out)
            
            # 残差连接 (⊕)
            x = x + residual
        
        return x


def test_bind_network():
    """测试 Bind Network 的功能"""
    print("测试 Bind Network...")
    
    # 创建模型
    model = BindNetwork(input_dim=1024, hidden_dim=4096, num_blocks=3)
    
    # 测试数据
    batch_size = 2
    seq_len = 1
    input_dim = 1024
    
    # 模拟 ImageBind 输出的图像特征
    image_feature = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"输入形状: {image_feature.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(image_feature)
    
    print(f"输出形状: {output.shape}")
    
    # 验证输出形状
    expected_shape = (batch_size, seq_len, 4096)
    assert output.shape == expected_shape, f"输出形状不匹配: {output.shape} vs {expected_shape}"
    
    print("✓ Bind Network 测试通过!")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return model


def create_simplified_bind_network():
    """
    创建简化版本的 Bind Network，更接近参考代码的结构
    """
    class SimplifiedBindNetwork(nn.Module):
        def __init__(self, input_dim: int = 1024, hidden_dim: int = 4096):
            super().__init__()
            
            # 投影层 (对应参考代码中的 image_bind_proj)
            self.proj = nn.Linear(input_dim, hidden_dim)
            
            # 三个重复块 (对应参考代码中的三个 image_bind_norm/f1/f2/f3)
            self.norm_1 = RMSNorm(hidden_dim)
            self.f1_1 = nn.Linear(hidden_dim, hidden_dim * 4)
            self.f2_1 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.f3_1 = nn.Linear(hidden_dim, hidden_dim * 4)
            
            self.norm_2 = RMSNorm(hidden_dim)
            self.f1_2 = nn.Linear(hidden_dim, hidden_dim * 4)
            self.f2_2 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.f3_2 = nn.Linear(hidden_dim, hidden_dim * 4)
            
            self.norm_3 = RMSNorm(hidden_dim)
            self.f1_3 = nn.Linear(hidden_dim, hidden_dim * 4)
            self.f2_3 = nn.Linear(hidden_dim * 4, hidden_dim)
            self.f3_3 = nn.Linear(hidden_dim, hidden_dim * 4)
        
        def forward(self, image_feature):
            # 投影到隐藏维度
            x = self.proj(image_feature)
            
            # 第一个块
            x_norm = self.norm_1(x)
            x = x + self.f2_1(F.silu(self.f1_1(x_norm)) * self.f3_1(x_norm))
            
            # 第二个块
            x_norm = self.norm_2(x)
            x = x + self.f2_2(F.silu(self.f1_2(x_norm)) * self.f3_2(x_norm))
            
            # 第三个块
            x_norm = self.norm_3(x)
            x = x + self.f2_3(F.silu(self.f1_3(x_norm)) * self.f3_3(x_norm))
            
            return x
    
    return SimplifiedBindNetwork


if __name__ == "__main__":
    print("=" * 60)
    print("Q3: Bind Network 实现测试")
    print("=" * 60)
    
    # 测试标准版本
    print("\n1. 测试标准 Bind Network:")
    model1 = test_bind_network()
    
    # 测试简化版本
    print("\n2. 测试简化 Bind Network:")
    SimplifiedBindNetwork = create_simplified_bind_network()
    model2 = SimplifiedBindNetwork(input_dim=1024, hidden_dim=4096)
    
    # 测试简化版本
    image_feature = torch.randn(2, 1, 1024)
    with torch.no_grad():
        output2 = model2(image_feature)
    
    print(f"简化版本输出形状: {output2.shape}")
    
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"简化版本参数数量: {total_params2:,}")
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！Bind Network 实现成功！")
    print("=" * 60)

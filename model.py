"""
扩散模型网络架构
实现基于U-Net的扩散模型，包含时间嵌入、残差块、注意力机制等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
import numpy as np

from config import get_config

class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，用于时间步嵌入"""
    
    def __init__(self, dim: int):
        """
        初始化正弦位置编码
        
        Args:
            dim: 编码维度
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            time: 时间步 [B]
            
        Returns:
            位置编码 [B, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        
        # 计算位置编码
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings

class TimeEmbedding(nn.Module):
    """时间嵌入模块"""
    
    def __init__(self, config=None):
        """
        初始化时间嵌入模块
        
        Args:
            config: 配置对象
        """
        super().__init__()
        self.config = config or get_config()
        
        # 时间嵌入维度
        time_dim = self.config.model.time_embed_dim
        
        # 正弦位置编码
        self.time_embed = SinusoidalPositionEmbedding(time_dim)
        
        # 时间MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            time: 时间步 [B]
            
        Returns:
            时间嵌入 [B, time_dim]
        """
        # 正弦位置编码
        time_emb = self.time_embed(time)
        
        # MLP处理
        time_emb = self.time_mlp(time_emb)
        
        return time_emb

class ResnetBlock(nn.Module):
    """残差块"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 dropout: float = 0.1, use_scale_shift_norm: bool = True):
        """
        初始化残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            time_emb_dim: 时间嵌入维度
            dropout: Dropout率
            use_scale_shift_norm: 是否使用缩放偏移归一化
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # 时间MLP
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2 if use_scale_shift_norm else out_channels)
        )
        
        # 第一个卷积块
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        # 第二个卷积块
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        # 输入输出通道数不同时的投影
        if self.in_channels != self.out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            time_emb: 时间嵌入 [B, time_dim]
            
        Returns:
            输出特征 [B, C, H, W]
        """
        # 时间嵌入处理
        time_emb = self.time_mlp(time_emb)
        
        # 第一个卷积块
        h = self.block1(x)
        
        # 时间嵌入注入
        if self.use_scale_shift_norm:
            # 缩放偏移归一化
            scale, shift = time_emb.chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        else:
            # 简单相加
            h = h + time_emb[:, :, None, None]
        
        # 第二个卷积块
        h = self.block2(h)
        
        # 残差连接
        return h + self.res_conv(x)

class AttentionBlock(nn.Module):
    """注意力块"""
    
    def __init__(self, channels: int, num_heads: int = 1, num_head_channels: int = -1):
        """
        初始化注意力块
        
        Args:
            channels: 通道数
            num_heads: 注意力头数
            num_head_channels: 每头通道数，-1表示自动计算
        """
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        if num_head_channels == -1:
            assert channels % num_heads == 0
            self.num_head_channels = channels // num_heads
        else:
            self.num_head_channels = num_head_channels
        
        # 归一化层
        self.norm = nn.GroupNorm(32, channels)
        
        # 多头注意力
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # 缩放因子
        self.scale = self.num_head_channels ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            
        Returns:
            输出特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 归一化
        h = self.norm(x)
        
        # 计算Q, K, V
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.num_head_channels, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, H*W, num_head_channels]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # 应用注意力
        h = torch.einsum('bhij,bhjd->bhid', attn, v)
        h = h.reshape(B, C, H, W)
        
        # 输出投影
        h = self.proj(h)
        
        return x + h

class Downsample(nn.Module):
    """下采样模块"""
    
    def __init__(self, channels: int, use_conv: bool = True):
        """
        初始化下采样模块
        
        Args:
            channels: 通道数
            use_conv: 是否使用卷积
        """
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        
        if use_conv:
            self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            
        Returns:
            下采样后的特征 [B, C, H//2, W//2]
        """
        return self.op(x)

class Upsample(nn.Module):
    """上采样模块"""
    
    def __init__(self, channels: int, use_conv: bool = True):
        """
        初始化上采样模块
        
        Args:
            channels: 通道数
            use_conv: 是否使用卷积
        """
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            
        Returns:
            上采样后的特征 [B, C, H*2, W*2]
        """
        # 最近邻上采样
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        if self.use_conv:
            x = self.conv(x)
        
        return x

class UNetModel(nn.Module):
    """U-Net扩散模型"""
    
    def __init__(self, config=None):
        """
        初始化U-Net模型
        
        Args:
            config: 配置对象
        """
        super().__init__()
        self.config = config or get_config()
        
        # 模型参数
        self.image_size = self.config.model.image_size
        self.in_channels = self.config.model.in_channels
        self.out_channels = self.config.model.out_channels
        self.model_channels = self.config.model.model_channels
        self.channel_mult = self.config.model.channel_mult
        self.num_res_blocks = self.config.model.num_res_blocks
        self.attention_resolutions = self.config.model.attention_resolutions
        self.dropout = self.config.model.dropout
        self.time_embed_dim = self.config.model.time_embed_dim
        self.num_heads = self.config.model.num_heads
        self.num_head_channels = self.config.model.num_head_channels
        self.use_scale_shift_norm = self.config.model.use_scale_shift_norm
        
        # 时间嵌入
        self.time_embed = TimeEmbedding(self.config)
        
        # 输入投影
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(self.in_channels, self.model_channels, 3, padding=1)
        ])
        
        # 计算通道数
        input_block_chans = [self.model_channels]
        ch = self.model_channels
        
        # 编码器（下采样路径）
        for level, mult in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                layers = [
                    ResnetBlock(
                        ch, mult * self.model_channels, self.time_embed_dim,
                        self.dropout, self.use_scale_shift_norm
                    )
                ]
                ch = mult * self.model_channels
                
                # 添加注意力层
                if ch in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, self.num_heads, self.num_head_channels)
                    )
                
                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)
            
            # 下采样（除了最后一个层级）
            if level != len(self.channel_mult) - 1:
                self.input_blocks.append(
                    nn.ModuleList([Downsample(ch)])
                )
                input_block_chans.append(ch)
        
        # 中间块
        self.middle_block = nn.ModuleList([
            ResnetBlock(
                ch, ch, self.time_embed_dim, self.dropout, self.use_scale_shift_norm
            ),
            AttentionBlock(ch, self.num_heads, self.num_head_channels),
            ResnetBlock(
                ch, ch, self.time_embed_dim, self.dropout, self.use_scale_shift_norm
            )
        ])
        
        # 解码器（上采样路径）
        self.output_blocks = nn.ModuleList([])
        
        # 计算需要的上采样次数
        # 从最深层开始，每个层级（除了第一个）都需要一次上采样
        # 总共需要 len(channel_mult) - 1 次上采样
        target_upsamples = len(self.channel_mult) - 1
        current_upsamples = 0
        
        for level, mult in list(enumerate(self.channel_mult))[::-1]:
            for i in range(self.num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResnetBlock(
                        ch + ich, mult * self.model_channels, self.time_embed_dim,
                        self.dropout, self.use_scale_shift_norm
                    )
                ]
                ch = mult * self.model_channels
                
                # 添加注意力层
                if ch in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, self.num_heads, self.num_head_channels)
                    )
                
                # 上采样：在每个层级的最后一个残差块后添加上采样
                # 但只在需要的时候添加，确保总上采样次数正确
                if level > 0 and i == self.num_res_blocks and current_upsamples < target_upsamples:
                    layers.append(Upsample(ch))
                    current_upsamples += 1
                
                self.output_blocks.append(nn.ModuleList(layers))
        
        # 输出层
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.out_channels, 3, padding=1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            timesteps: 时间步 [B]
            
        Returns:
            预测的噪声 [B, C, H, W]
        """
        # 时间嵌入
        time_emb = self.time_embed(timesteps)
        
        # 输入特征
        hs = []
        h = x
        
        # 编码器路径
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResnetBlock):
                        h = layer(h, time_emb)
                    elif isinstance(layer, AttentionBlock):
                        h = layer(h)
                    elif isinstance(layer, Downsample):
                        h = layer(h)
                hs.append(h)
            else:
                h = module(h)
                hs.append(h)
        
        # 中间块
        for module in self.middle_block:
            if isinstance(module, ResnetBlock):
                h = module(h, time_emb)
            elif isinstance(module, AttentionBlock):
                h = module(h)
        
        # 解码器路径
        for module in self.output_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResnetBlock):
                        # 获取跳跃连接的特征
                        skip_feature = hs.pop()
                        # 确保特征尺寸匹配
                        if skip_feature.shape[2:] != h.shape[2:]:
                            skip_feature = torch.nn.functional.interpolate(
                                skip_feature, size=h.shape[2:], mode='nearest'
                            )
                        h = torch.cat([h, skip_feature], dim=1)
                        h = layer(h, time_emb)
                    elif isinstance(layer, AttentionBlock):
                        h = layer(h)
                    elif isinstance(layer, Upsample):
                        h = layer(h)
        
        # 输出
        return self.out(h)

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    """测试模型"""
    print("=== 测试U-Net模型 ===")
    
    # 获取配置
    config = get_config()
    
    # 创建模型
    model = UNetModel(config)
    
    # 打印模型信息
    print(f"模型参数数量: {count_parameters(model):,}")
    print(f"输入通道数: {model.in_channels}")
    print(f"输出通道数: {model.out_channels}")
    print(f"基础通道数: {model.model_channels}")
    print(f"通道倍数: {model.channel_mult}")
    print(f"残差块数量: {model.num_res_blocks}")
    print(f"注意力分辨率: {model.attention_resolutions}")
    
    # 测试前向传播
    batch_size = 4
    x = torch.randn(batch_size, config.model.in_channels, 
                    config.model.image_size, config.model.image_size)
    timesteps = torch.randint(0, config.diffusion.num_timesteps, (batch_size,))
    
    print(f"\n输入形状: {x.shape}")
    print(f"时间步形状: {timesteps.shape}")
    
    # 前向传播
    with torch.no_grad():
        output = model(x, timesteps)
        print(f"输出形状: {output.shape}")
        print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n模型测试完成！")

if __name__ == "__main__":
    main() 
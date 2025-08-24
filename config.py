"""
扩散模型配置文件
包含所有模型架构、训练参数和系统配置
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ModelConfig:
    """模型架构配置"""
    # 图像配置
    image_size: int = 32  # CIFAR-10图像尺寸
    in_channels: int = 3   # RGB三通道
    out_channels: int = 3  # 输出通道数
    
    # 网络架构
    model_channels: int = 64      # 基础通道数
    channel_mult: Tuple[int, ...] = (1, 2, 4, 8)  # 通道倍数
    num_res_blocks: int = 2       # 残差块数量
    attention_resolutions: Tuple[int, ...] = ()  # 注意力层分辨率，留空自动计算
    dropout: float = 0.1          # Dropout率
    use_fp16: bool = False        # 是否使用混合精度
    
    # 时间嵌入
    time_embed_dim: int = 512     # 时间嵌入维度
    time_embed_act: str = "swish" # 激活函数
    
    # 注意力机制
    num_heads: int = 8            # 多头注意力头数
    num_head_channels: int = -1   # 每头通道数 (-1表示自动计算)
    use_scale_shift_norm: bool = True  # 是否使用缩放偏移归一化

@dataclass
class DiffusionConfig:
    """扩散过程配置"""
    # 噪声调度
    num_timesteps: int = 1000     # 总时间步数
    beta_start: float = 1e-4      # 初始β值
    beta_end: float = 0.02        # 最终β值
    schedule_type: str = "linear" # 调度类型: linear, cosine
    
    # 采样配置
    sample_steps: int = 1000      # 采样步数
    sample_method: str = "ddpm"   # 采样方法: ddpm, ddim
    eta: float = 0.0              # DDIM采样参数
    
    # 损失函数
    loss_type: str = "mse"        # 损失类型: mse, l1, huber
    loss_weight: float = 1.0      # 损失权重

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础参数
    batch_size: int = 64          # 批次大小
    num_epochs: int = 100        # 训练轮数
    learning_rate: float = 2e-4   # 学习率
    weight_decay: float = 1e-4    # 权重衰减
    
    # 优化器
    optimizer: str = "adam"       # 优化器类型
    beta1: float = 0.9           # Adam优化器β1
    beta2: float = 0.999         # Adam优化器β2
    eps: float = 1e-8            # 数值稳定性参数
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # 学习率调度器
    warmup_steps: int = 5000      # 预热步数
    min_lr: float = 1e-6         # 最小学习率
    
    # 训练策略
    gradient_clip: float = 1.0    # 梯度裁剪
    accumulation_steps: int = 1    # 梯度累积步数
    mixed_precision: bool = False # 混合精度训练
    
    # 保存和日志
    save_every: int = 100         # 每多少轮保存一次
    log_every: int = 10           # 每多少步记录一次
    eval_every: int = 100         # 每多少轮评估一次

@dataclass
class DataConfig:
    """数据配置"""
    # 数据集
    dataset_name: str = "cifar10" # 数据集名称
    data_dir: str = "./data"      # 数据目录
    num_workers: int = 4          # 数据加载器工作进程数
    pin_memory: bool = True       # 是否使用固定内存
    
    # 数据预处理
    normalize: bool = True        # 是否归一化
    normalize_range: Tuple[float, float] = (-1.0, 1.0)  # 归一化范围
    augment: bool = True          # 是否使用数据增强
    horizontal_flip: bool = True  # 随机水平翻转

@dataclass
class SystemConfig:
    """系统配置"""
    # 设备
    device: str = "auto"          # 设备选择: auto, cuda, cpu
    seed: int = 42               # 随机种子
    
    # 路径
    checkpoint_dir: str = "./checkpoints"  # 检查点目录
    result_dir: str = "./results"          # 结果目录
    log_dir: str = "./logs"               # 日志目录
    
    # 实验
    experiment_name: str = "diffusion_experiment"  # 实验名称
    resume: bool = False         # 是否从检查点恢复
    resume_path: str = ""        # 恢复检查点路径

@dataclass
class Config:
    """总配置类"""
    model: ModelConfig = ModelConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    system: SystemConfig = SystemConfig()
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        os.makedirs(self.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.system.result_dir, exist_ok=True)
        os.makedirs(self.system.log_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)
        
        # 设置设备
        if self.system.device == "auto":
            import torch
            self.system.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 计算注意力分辨率
        if not self.model.attention_resolutions:
            self.model.attention_resolutions = tuple(
                self.model.image_size // (2 ** i) 
                for i in range(len(self.model.channel_mult))
            )

# 默认配置实例
config = Config()

def get_config() -> Config:
    """获取配置实例"""
    return config

def update_config(**kwargs):
    """更新配置参数"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # 尝试更新子配置
            for sub_config_name in ['model', 'diffusion', 'training', 'data', 'system']:
                if hasattr(config, sub_config_name):
                    sub_config = getattr(config, sub_config_name)
                    if hasattr(sub_config, key):
                        setattr(sub_config, key, value)
                        break

if __name__ == "__main__":
    # 打印配置信息
    print("=== 扩散模型配置 ===")
    print(f"模型配置: {config.model}")
    print(f"扩散配置: {config.diffusion}")
    print(f"训练配置: {config.training}")
    print(f"数据配置: {config.data}")
    print(f"系统配置: {config.system}") 
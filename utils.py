"""
工具函数模块
包含图像处理、可视化、检查点保存等实用功能
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from typing import List, Tuple, Optional, Dict, Any
import time
from datetime import datetime
import logging
from pathlib import Path

from config import get_config

def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    设置日志记录
    
    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # 配置日志记录器
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int, loss: float, config: Any, 
                   checkpoint_dir: str, filename: str) -> str:
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮数
        loss: 当前损失
        config: 配置对象
        checkpoint_dir: 检查点目录
        filename: 文件名
        
    Returns:
        保存的文件路径
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 准备检查点数据
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    # 添加调度器状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 保存检查点
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器
        
    Returns:
        检查点信息字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def save_images(images: torch.Tensor, save_dir: str, filename: str, 
                normalize: bool = True, save_individual: bool = False) -> str:
    """
    保存图像
    
    Args:
        images: 图像张量 [B, C, H, W]
        save_dir: 保存目录
        filename: 文件名
        normalize: 是否归一化到[0, 1]
        save_individual: 是否保存单独的图像
        
    Returns:
        保存的文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy数组
    if images.dim() == 4:
        images_np = images.detach().cpu().numpy()
    else:
        images_np = images.detach().cpu().numpy()[None, ...]
    
    # 归一化到[0, 1]
    if normalize:
        if images_np.min() < 0:
            images_np = (images_np + 1) / 2
        images_np = np.clip(images_np, 0, 1)
    
    # 调整维度顺序 [B, C, H, W] -> [B, H, W, C]
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    
    # 保存网格图像
    grid_path = os.path.join(save_dir, f"{filename}_grid.png")
    save_image_grid(images_np, grid_path)
    
    # 保存单独图像
    if save_individual:
        individual_dir = os.path.join(save_dir, f"{filename}_individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for i, img in enumerate(images_np):
            img_path = os.path.join(individual_dir, f"{filename}_{i:03d}.png")
            save_single_image(img, img_path)
    
    return grid_path

def save_image_grid(images: np.ndarray, save_path: str, nrow: int = 8) -> None:
    """
    保存图像网格
    
    Args:
        images: 图像数组 [B, H, W, C]
        save_path: 保存路径
        nrow: 每行图像数量
    """
    n_images = len(images)
    ncol = (n_images + nrow - 1) // nrow
    
    # 创建图像网格
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 2, ncol * 2))
    
    if ncol == 1:
        axes = axes[None, :]
    if nrow == 1:
        axes = axes[:, None]
    
    # 填充图像
    for i in range(n_images):
        row = i // nrow
        col = i % nrow
        axes[row, col].imshow(images[i])
        axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(n_images, nrow * ncol):
        row = i // nrow
        col = i % nrow
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_single_image(image: np.ndarray, save_path: str) -> None:
    """
    保存单张图像
    
    Args:
        image: 图像数组 [H, W, C]
        save_path: 保存路径
    """
    # 转换为PIL图像
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image)
    pil_image.save(save_path)

def create_denoising_animation(intermediates: List[torch.Tensor], save_dir: str, 
                              filename: str, fps: int = 10) -> str:
    """
    创建去噪过程动画
    
    Args:
        intermediates: 中间结果列表
        save_dir: 保存目录
        filename: 文件名
        fps: 帧率
        
    Returns:
        保存的动画文件路径
    """
    try:
        import imageio
    except ImportError:
        print("需要安装imageio来创建动画")
        return ""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy数组并归一化
    frames = []
    for img in intermediates:
        img_np = img.detach().cpu().numpy()
        if img_np.min() < 0:
            img_np = (img_np + 1) / 2
        img_np = np.clip(img_np, 0, 1)
        
        # 转换为uint8
        img_np = (img_np * 255).astype(np.uint8)
        
        # 调整维度 [B, C, H, W] -> [B, H, W, C]
        img_np = np.transpose(img_np, (0, 2, 3, 1))
        
        # 取第一张图像
        frames.append(img_np[0])
    
    # 保存动画
    animation_path = os.path.join(save_dir, f"{filename}_animation.gif")
    imageio.mimsave(animation_path, frames, fps=fps)
    
    return animation_path

def plot_training_curves(train_losses: List[float], val_losses: Optional[List[float]] = None,
                        save_path: Optional[str] = None, title: str = "训练曲线") -> None:
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
        title: 图表标题
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
    
    # 绘制验证损失
    if val_losses is not None:
        plt.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel('轮数', fontsize=12)
    plt.ylabel('损失', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 设置y轴范围
    if train_losses:
        y_min = min(train_losses) * 0.9
        y_max = max(train_losses) * 1.1
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_learning_rate_schedule(optimizer: torch.optim.Optimizer, 
                               scheduler: torch.optim.lr_scheduler._LRScheduler,
                               num_steps: int, save_path: Optional[str] = None) -> None:
    """
    绘制学习率调度曲线
    
    Args:
        optimizer: 优化器
        scheduler: 学习率调度器
        num_steps: 总步数
        save_path: 保存路径
    """
    lrs = []
    steps = []
    
    # 模拟学习率变化
    for step in range(num_steps):
        lrs.append(optimizer.param_groups[0]['lr'])
        steps.append(step)
        scheduler.step()
    
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lrs, 'b-', linewidth=2)
    plt.title('学习率调度', fontsize=16)
    plt.xlabel('步数', fontsize=12)
    plt.ylabel('学习率', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习率调度曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_denoising_process(model: nn.Module, diffusion: Any, 
                               x_start: torch.Tensor, save_dir: str, 
                               filename: str, num_steps: int = 10) -> str:
    """
    可视化去噪过程
    
    Args:
        model: 扩散模型
        diffusion: 扩散过程对象
        x_start: 原始图像
        save_dir: 保存目录
        filename: 文件名
        num_steps: 可视化步数
        
    Returns:
        保存的文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取设备信息
    device = next(model.parameters()).device
    
    # 确保x_start在正确的设备上
    x_start = x_start.to(device)
    
    # 确保扩散过程在正确的设备上
    diffusion_device = diffusion.device
    if device != diffusion_device:
        print(f"警告：模型设备 ({device}) 与扩散过程设备 ({diffusion_device}) 不匹配")
        print(f"将扩散过程移动到模型设备: {device}")
        # 将扩散过程移动到模型设备
        for key in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                   'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                   'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod',
                   'posterior_variance', 'posterior_log_variance_clipped',
                   'posterior_mean_coef1', 'posterior_mean_coef2']:
            if hasattr(diffusion.noise_scheduler, key):
                tensor = getattr(diffusion.noise_scheduler, key)
                setattr(diffusion.noise_scheduler, key, tensor.to(device))
        diffusion.device = device
    
    # 生成去噪过程
    with torch.no_grad():
        intermediates = diffusion.p_sample_loop(
            model, x_start.shape, return_intermediates=True
        )
    
    if isinstance(intermediates, tuple):
        x_final, intermediates = intermediates
    else:
        x_final = intermediates
        intermediates = [x_final]
    
    # 添加原始图像，确保所有图像都在同一设备上
    all_images = [x_start] + [img.to(device) for img in intermediates]
    
    # 选择要可视化的图像
    step_indices = np.linspace(0, len(all_images) - 1, num_steps, dtype=int)
    selected_images = [all_images[i] for i in step_indices]
    
    # 保存图像网格，先移动到CPU再堆叠
    save_path = os.path.join(save_dir, f"{filename}_denoising_process.png")
    selected_images_cpu = [img.cpu() for img in selected_images]
    
    # 堆叠图像并调整维度 [B, C, H, W] -> [B, H, W, C]
    stacked_images = torch.stack(selected_images_cpu)
    
    # 确保张量是4维的 [B, C, H, W]
    if stacked_images.dim() == 5:
        # 如果是5维，需要完全展平批次维度
        # 形状 [5, 1, 3, 32, 32] -> [5, 3, 32, 32]
        stacked_images = stacked_images.squeeze(1)
    
    # 调整维度 [B, C, H, W] -> [B, H, W, C]
    stacked_images = stacked_images.permute(0, 2, 3, 1)
    
    # 转换为numpy数组并归一化
    images_np = stacked_images.numpy()
    if images_np.min() < 0:
        images_np = (images_np + 1) / 2
    images_np = np.clip(images_np, 0, 1)
    
    save_image_grid(images_np, save_path, nrow=num_steps)
    
    return save_path

def calculate_fid_score(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """
    计算FID分数（简化版本）
    
    Args:
        real_images: 真实图像
        fake_images: 生成图像
        
    Returns:
        FID分数
    """
    # 这里实现一个简化的FID计算
    # 实际应用中建议使用完整的FID实现
    
    # 计算均值和协方差
    real_mean = real_images.mean(dim=0)
    fake_mean = fake_images.mean(dim=0)
    
    real_cov = torch.cov(real_images.reshape(real_images.shape[0], -1).T)
    fake_cov = torch.cov(fake_images.reshape(fake_images.shape[0], -1).T)
    
    # 计算FID
    mean_diff = real_mean - fake_mean
    mean_diff_sq = torch.sum(mean_diff ** 2)
    
    cov_diff = real_cov + fake_cov - 2 * torch.sqrt(real_cov * fake_cov)
    cov_diff = torch.trace(cov_diff)
    
    fid = mean_diff_sq + cov_diff
    
    return fid.item()

def set_random_seed(seed: int) -> None:
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息
    
    Returns:
        设备信息字典
    """
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    
    if torch.cuda.is_available():
        device_info['memory_allocated'] = torch.cuda.memory_allocated(0)
        device_info['memory_reserved'] = torch.cuda.memory_reserved(0)
        device_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
    
    return device_info

def print_device_info() -> None:
    """打印设备信息"""
    device_info = get_device_info()
    
    print("=== 设备信息 ===")
    print(f"CUDA可用: {device_info['cuda_available']}")
    print(f"设备数量: {device_info['device_count']}")
    print(f"当前设备: {device_info['current_device']}")
    print(f"设备名称: {device_info['device_name']}")
    
    if device_info['cuda_available']:
        print(f"已分配内存: {device_info['memory_allocated'] / 1024**3:.2f} GB")
        print(f"已保留内存: {device_info['memory_reserved'] / 1024**3:.2f} GB")
        print(f"总内存: {device_info['memory_total'] / 1024**3:.2f} GB")

def main():
    """测试工具函数"""
    print("=== 测试工具函数 ===")
    
    # 获取配置
    config = get_config()
    
    # 设置随机种子
    set_random_seed(config.system.seed)
    
    # 打印设备信息
    print_device_info()
    
    # 测试图像保存
    print("\n=== 测试图像保存 ===")
    test_images = torch.randn(16, 3, 32, 32)
    save_path = save_images(test_images, config.system.result_dir, "test_images")
    print(f"测试图像已保存到: {save_path}")
    
    # 测试训练曲线绘制
    print("\n=== 测试训练曲线绘制 ===")
    train_losses = [0.5, 0.4, 0.3, 0.25, 0.2, 0.18, 0.16, 0.15, 0.14, 0.13]
    val_losses = [0.6, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24, 0.23]
    
    plot_path = os.path.join(config.system.result_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, plot_path)
    
    print("\n工具函数测试完成！")

if __name__ == "__main__":
    main() 
"""
扩散过程核心实现
包含前向过程、逆向过程、噪声调度和采样策略
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Union
import matplotlib.pyplot as plt
import os

from config import get_config

class NoiseScheduler:
    """噪声调度器，管理β值和相关参数"""
    
    def __init__(self, config=None):
        """
        初始化噪声调度器
        
        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        self.num_timesteps = self.config.diffusion.num_timesteps
        self.beta_start = self.config.diffusion.beta_start
        self.beta_end = self.config.diffusion.beta_end
        self.schedule_type = self.config.diffusion.schedule_type
        
        # 计算β值序列
        self.betas = self._get_beta_schedule()
        
        # 预计算相关参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 计算q(xₜ|x₀)分布的参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 计算q(xₜ₋₁|xₜ, x₀)分布的参数
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 计算后验方差
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas_cumprod) / (1.0 - self.alphas_cumprod)
        )
        
        # 初始化时所有张量都在CPU上，稍后会被移动到正确的设备
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """
        获取β值调度
        
        Returns:
            β值序列
        """
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.schedule_type == "cosine":
            # 余弦调度，来自Improved DDPM论文
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0, 0.999)
        else:
            raise ValueError(f"不支持的调度类型: {self.schedule_type}")
    
    def add_noise(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        向原始图像添加噪声
        
        Args:
            x_start: 原始图像 [B, C, H, W]
            t: 时间步 [B]
            noise: 预定义的噪声，如果为None则随机生成
            
        Returns:
            添加噪声后的图像和噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 获取对应时间步的参数
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        # 计算q(xₜ|x₀)
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算q(xₜ₋₁|xₜ, x₀)的后验均值和方差
        
        Args:
            x_start: 原始图像
            x_t: 时间步t的图像
            t: 时间步
            
        Returns:
            后验均值、方差和log方差
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, model, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> dict:
        """
        计算p(xₜ₋₁|xₜ)的均值和方差
        
        Args:
            model: 扩散模型
            x: 输入图像
            t: 时间步
            clip_denoised: 是否裁剪去噪结果
            
        Returns:
            包含均值、方差等信息的字典
        """
        # 预测噪声
        predicted_noise = model(x, t)
        
        # 计算x₀的预测值
        alpha_cumprod_t = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        x_start_pred = (x - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
        
        if clip_denoised:
            x_start_pred = torch.clamp(x_start_pred, -1, 1)
        
        # 计算后验参数
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start_pred, x, t
        )
        
        return {
            'mean': model_mean,
            'variance': posterior_variance,
            'log_variance': posterior_log_variance,
            'pred_xstart': x_start_pred,
            'pred_noise': predicted_noise
        }
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        随机采样时间步
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            随机时间步
        """
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()

class DiffusionProcess:
    """扩散过程主类"""
    
    def __init__(self, config=None):
        """
        初始化扩散过程
        
        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        
        # 确定设备
        if self.config.system.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.system.device)
        
        self.noise_scheduler = NoiseScheduler(config)
        
        # 将噪声调度器移动到设备
        for key in ['betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
                   'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod',
                   'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod',
                   'posterior_variance', 'posterior_log_variance_clipped',
                   'posterior_mean_coef1', 'posterior_mean_coef2']:
            if hasattr(self.noise_scheduler, key):
                tensor = getattr(self.noise_scheduler, key)
                setattr(self.noise_scheduler, key, tensor.to(self.device))
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        从q(xₜ|x₀)采样
        
        Args:
            x_start: 原始图像
            t: 时间步
            noise: 预定义噪声
            
        Returns:
            采样结果
        """
        return self.noise_scheduler.add_noise(x_start, t, noise)[0]
    
    def p_losses(self, model, x_start: torch.Tensor, t: Optional[torch.Tensor] = None) -> dict:
        """
        计算训练损失
        
        Args:
            model: 扩散模型
            x_start: 原始图像
            t: 时间步，如果为None则随机采样
            
        Returns:
            损失信息字典
        """
        if t is None:
            t = self.noise_scheduler.sample_timesteps(x_start.shape[0], self.device)
        
        # 添加噪声
        x_t, noise = self.noise_scheduler.add_noise(x_start, t)
        
        # 预测噪声
        predicted_noise = model(x_t, t)
        
        # 计算损失
        if self.config.diffusion.loss_type == "mse":
            loss = F.mse_loss(predicted_noise, noise, reduction='none')
        elif self.config.diffusion.loss_type == "l1":
            loss = F.l1_loss(predicted_noise, noise, reduction='none')
        elif self.config.diffusion.loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_noise, noise, reduction='none')
        else:
            raise ValueError(f"不支持的损失类型: {self.config.diffusion.loss_type}")
        
        # 计算平均损失
        loss = loss.mean(dim=[1, 2, 3]).mean()
        
        return {
            'loss': loss,
            'pred_noise': predicted_noise,
            'target_noise': noise,
            'x_t': x_t,
            't': t
        }
    
    @torch.no_grad()
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        """
        从p(xₜ₋₁|xₜ)采样一步
        
        Args:
            model: 扩散模型
            x: 输入图像
            t: 时间步
            clip_denoised: 是否裁剪去噪结果
            
        Returns:
            采样结果
        """
        # 获取模型预测
        out = self.noise_scheduler.p_mean_variance(model, x, t, clip_denoised)
        
        # 采样
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        
        sample = out['mean'] + nonzero_mask * torch.sqrt(out['variance']) * noise
        
        return sample
    
    @torch.no_grad()
    def p_sample_loop(self, model, shape: Tuple[int, ...], clip_denoised: bool = True, 
                     return_intermediates: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        完整的采样循环
        
        Args:
            model: 扩散模型
            shape: 输出形状
            clip_denoised: 是否裁剪去噪结果
            return_intermediates: 是否返回中间结果
            
        Returns:
            采样结果，如果return_intermediates为True则返回中间结果列表
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        intermediates = []
        
        # 反向时间步
        time_steps = list(range(self.config.diffusion.num_timesteps))[::-1]
        
        for i, t in enumerate(time_steps):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_tensor, clip_denoised)
            
            if return_intermediates and i % (self.config.diffusion.num_timesteps // 10) == 0:
                intermediates.append(x.cpu())
        
        if return_intermediates:
            return x, intermediates
        return x
    
    @torch.no_grad()
    def sample(self, model, batch_size: int = 16, return_intermediates: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        生成样本
        
        Args:
            model: 扩散模型
            batch_size: 批次大小
            return_intermediates: 是否返回中间结果
            
        Returns:
            生成的样本
        """
        shape = (batch_size, self.config.model.in_channels, 
                self.config.model.image_size, self.config.model.image_size)
        
        return self.p_sample_loop(model, shape, return_intermediates=return_intermediates)
    
    def visualize_noise_schedule(self, save_path: Optional[str] = None):
        """
        可视化噪声调度
        
        Args:
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # β值
        axes[0, 0].plot(self.noise_scheduler.betas.cpu().numpy())
        axes[0, 0].set_title('β值调度')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('β值')
        axes[0, 0].grid(True)
        
        # α累积积
        axes[0, 1].plot(self.noise_scheduler.alphas_cumprod.cpu().numpy())
        axes[0, 1].set_title('α累积积')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('α累积积')
        axes[0, 1].grid(True)
        
        # 噪声标准差
        axes[1, 0].plot(self.noise_scheduler.sqrt_one_minus_alphas_cumprod.cpu().numpy())
        axes[1, 0].set_title('噪声标准差')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('标准差')
        axes[1, 0].grid(True)
        
        # 信号强度
        axes[1, 1].plot(self.noise_scheduler.sqrt_alphas_cumprod.cpu().numpy())
        axes[1, 1].set_title('信号强度')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('信号强度')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"噪声调度可视化已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()

def main():
    """测试扩散过程"""
    print("=== 测试扩散过程 ===")
    
    # 获取配置
    config = get_config()
    
    # 创建扩散过程
    diffusion = DiffusionProcess(config)
    
    # 可视化噪声调度
    save_path = os.path.join(config.system.result_dir, "noise_schedule.png")
    diffusion.visualize_noise_schedule(save_path)
    
    print("扩散过程测试完成！")

if __name__ == "__main__":
    main() 
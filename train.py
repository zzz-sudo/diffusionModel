"""
扩散模型训练脚本
包含完整的训练循环、断点续传、可视化等功能
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any

from config import get_config
from dataset import CIFAR10Dataset
from model import UNetModel
from diffusion import DiffusionProcess
from utils import (
    setup_logging, save_checkpoint, load_checkpoint, save_images,
    plot_training_curves, plot_learning_rate_schedule, set_random_seed,
    print_device_info, visualize_denoising_process
)

class DiffusionTrainer:
    """扩散模型训练器"""
    
    def __init__(self, config=None):
        """
        初始化训练器
        
        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        
        # 设置随机种子
        set_random_seed(self.config.system.seed)
        
        # 设置设备
        self.device = torch.device(self.config.system.device)
        
        # 设置日志
        self.logger = setup_logging(
            self.config.system.log_dir, 
            self.config.system.experiment_name
        )
        
        # 创建目录
        os.makedirs(self.config.system.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.system.result_dir, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        # 训练状态
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        self.logger.info("训练器初始化完成")
    
    def _init_components(self):
        """初始化训练组件"""
        # 数据集
        self.logger.info("初始化数据集...")
        self.dataset_manager = CIFAR10Dataset(self.config)
        self.train_loader, self.val_loader = self.dataset_manager.get_dataloaders()
        
        # 模型
        self.logger.info("初始化模型...")
        self.model = UNetModel(self.config).to(self.device)
        
        # 扩散过程
        self.logger.info("初始化扩散过程...")
        self.diffusion = DiffusionProcess(self.config)
        
        # 优化器
        self.logger.info("初始化优化器...")
        if self.config.training.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=(self.config.training.beta1, self.config.training.beta2),
                eps=self.config.training.eps,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {self.config.training.optimizer}")
        
        # 学习率调度器
        if self.config.training.lr_scheduler.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        else:
            self.scheduler = None
        
        # 损失函数
        if self.config.diffusion.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.config.diffusion.loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.config.diffusion.loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"不支持的损失类型: {self.config.diffusion.loss_type}")
        
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            是否成功加载
        """
        try:
            self.logger.info(f"加载检查点: {checkpoint_path}")
            
            checkpoint = load_checkpoint(
                checkpoint_path, 
                self.model, 
                self.optimizer, 
                self.scheduler
            )
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint.get('loss', float('inf'))
            
            self.logger.info(f"从轮数 {self.start_epoch} 恢复训练")
            self.logger.info(f"最佳损失: {self.best_loss:.6f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"加载检查点失败: {e}")
            return False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个轮次
        
        Args:
            epoch: 当前轮数
            
        Returns:
            训练统计信息
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            # 移动数据到设备
            images = images.to(self.device)
            
            # 前向传播
            loss_dict = self.diffusion.p_losses(self.model, images)
            loss = loss_dict['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip
                )
            
            # 优化器步进
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.6f}",
                'Avg Loss': f"{total_loss / (batch_idx + 1):.6f}"
            })
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
        
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        验证一个轮次
        
        Args:
            epoch: 当前轮数
            
        Returns:
            验证统计信息
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, _ in tqdm(self.val_loader, desc=f"Validation {epoch}"):
                # 移动数据到设备
                images = images.to(self.device)
                
                # 前向传播
                loss_dict = self.diffusion.p_losses(self.model, images)
                loss = loss_dict['loss']
                
                # 统计
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        return {'val_loss': avg_loss}
    
    def save_samples(self, epoch: int, num_samples: int = 16):
        """
        保存生成的样本
        
        Args:
            epoch: 当前轮数
            num_samples: 样本数量
        """
        self.logger.info(f"生成样本 (轮数 {epoch})...")
        
        # 生成样本
        with torch.no_grad():
            samples = self.diffusion.sample(
                self.model, 
                batch_size=num_samples,
                return_intermediates=False
            )
        
        # 保存样本
        save_path = save_images(
            samples, 
            self.config.system.result_dir, 
            f"epoch_{epoch:04d}_samples"
        )
        
        self.logger.info(f"样本已保存到: {save_path}")
        
        # 可视化去噪过程（每100轮）
        if epoch % 100 == 0:
            self.logger.info("可视化去噪过程...")
            
            # 获取一个真实样本
            real_images, _ = next(iter(self.val_loader))
            real_images = real_images[:1].to(self.device)
            
            # 可视化去噪过程
            denoise_path = visualize_denoising_process(
                self.model, 
                self.diffusion, 
                real_images, 
                self.config.system.result_dir, 
                f"epoch_{epoch:04d}_denoising"
            )
            
            self.logger.info(f"去噪过程可视化已保存到: {denoise_path}")
    
    def train(self, resume_path: Optional[str] = None):
        """
        开始训练
        
        Args:
            resume_path: 恢复训练的检查点路径
        """
        self.logger.info("开始训练...")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"总轮数: {self.config.training.num_epochs}")
        self.logger.info(f"批次大小: {self.config.training.batch_size}")
        self.logger.info(f"学习率: {self.config.training.learning_rate}")
        
        # 打印设备信息
        print_device_info()
        
        # 恢复训练
        if resume_path and os.path.exists(resume_path):
            if not self.load_checkpoint(resume_path):
                self.logger.warning("恢复训练失败，从头开始训练")
        
        # 训练循环
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_stats = self.train_epoch(epoch)
            
            # 验证
            val_stats = self.validate_epoch(epoch)
            
            # 统计
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录损失
            self.train_losses.append(train_stats['train_loss'])
            self.val_losses.append(val_stats['val_loss'])
            
            # 日志
            self.logger.info(
                f"Epoch {epoch:04d}/{self.config.training.num_epochs:04d} - "
                f"Train Loss: {train_stats['train_loss']:.6f}, "
                f"Val Loss: {val_stats['val_loss']:.6f}, "
                f"LR: {current_lr:.2e}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 保存检查点
            if epoch % self.config.training.save_every == 0:
                checkpoint_filename = f"checkpoint_epoch_{epoch:04d}.pth"
                checkpoint_path = save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    self.scheduler, 
                    epoch, 
                    val_stats['val_loss'], 
                    self.config, 
                    self.config.system.checkpoint_dir, 
                    checkpoint_filename
                )
                self.logger.info(f"检查点已保存: {checkpoint_path}")
                
                # 保存最佳模型
                if val_stats['val_loss'] < self.best_loss:
                    self.best_loss = val_stats['val_loss']
                    best_checkpoint_path = save_checkpoint(
                        self.model, 
                        self.optimizer, 
                        self.scheduler, 
                        epoch, 
                        val_stats['val_loss'], 
                        self.config, 
                        self.config.system.checkpoint_dir, 
                        "best_model.pth"
                    )
                    self.logger.info(f"最佳模型已保存: {best_checkpoint_path}")
            
            # 生成样本
            if epoch % self.config.training.eval_every == 0:
                self.save_samples(epoch)
            
            # 绘制训练曲线
            if epoch % 50 == 0:
                plot_path = os.path.join(
                    self.config.system.result_dir, 
                    f"training_curves_epoch_{epoch:04d}.png"
                )
                plot_training_curves(
                    self.train_losses, 
                    self.val_losses, 
                    plot_path, 
                    f"训练曲线 (轮数 {epoch})"
                )
        
        # 训练完成
        self.logger.info("训练完成！")
        
        # 保存最终模型
        final_checkpoint_path = save_checkpoint(
            self.model, 
            self.optimizer, 
            self.scheduler, 
            self.config.training.num_epochs - 1, 
            self.val_losses[-1], 
            self.config, 
            self.config.system.checkpoint_dir, 
            "final_model.pth"
        )
        self.logger.info(f"最终模型已保存: {final_checkpoint_path}")
        
        # 绘制最终训练曲线
        final_plot_path = os.path.join(
            self.config.system.result_dir, 
            "final_training_curves.png"
        )
        plot_training_curves(
            self.train_losses, 
            self.val_losses, 
            final_plot_path, 
            "最终训练曲线"
        )
        
        # 绘制学习率调度
        if self.scheduler is not None:
            lr_plot_path = os.path.join(
                self.config.system.result_dir, 
                "learning_rate_schedule.png"
            )
            plot_learning_rate_schedule(
                self.optimizer, 
                self.scheduler, 
                self.config.training.num_epochs, 
                lr_plot_path
            )
        
        # 生成最终样本
        self.save_samples(self.config.training.num_epochs - 1, num_samples=64)
        
        self.logger.info("所有结果已保存完成！")

def main():
    """主函数"""
    print("=== 扩散模型训练 ===")
    
    # 获取配置
    config = get_config()
    
    # 创建训练器
    trainer = DiffusionTrainer(config)
    
    # 检查是否恢复训练
    resume_path = ""
    if config.system.resume:
        if config.system.resume_path:
            resume_path = config.system.resume_path
        else:
            # 查找最新的检查点
            checkpoint_dir = config.system.checkpoint_dir
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
                if checkpoints:
                    # 按文件名排序，找到最新的
                    checkpoints.sort()
                    resume_path = os.path.join(checkpoint_dir, checkpoints[-1])
                    print(f"找到检查点: {resume_path}")
    
    # 开始训练
    trainer.train(resume_path=resume_path if resume_path else None)

if __name__ == "__main__":
    main() 
"""
扩散模型采样脚本
用于生成图像和可视化去噪过程
"""

import os
import torch
import argparse
from typing import Optional, List
import numpy as np

from config import get_config
from model import UNetModel
from diffusion import DiffusionProcess
from utils import (
    load_checkpoint, save_images, visualize_denoising_process,
    create_denoising_animation, set_random_seed
)

class DiffusionSampler:
    """扩散模型采样器"""
    
    def __init__(self, config=None):
        """
        初始化采样器
        
        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        
        # 设置随机种子
        set_random_seed(self.config.system.seed)
        
        # 设置设备
        self.device = torch.device(self.config.system.device)
        
        # 初始化模型
        self.model = UNetModel(self.config).to(self.device)
        
        # 初始化扩散过程
        self.diffusion = DiffusionProcess(self.config)
        
        print(f"采样器初始化完成，设备: {self.device}")
    
    def load_model(self, checkpoint_path: str) -> bool:
        """
        加载训练好的模型
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            是否成功加载
        """
        try:
            print(f"加载模型: {checkpoint_path}")
            
            checkpoint = load_checkpoint(checkpoint_path, self.model)
            
            print(f"模型加载成功，训练轮数: {checkpoint['epoch']}")
            print(f"训练损失: {checkpoint['loss']:.6f}")
            
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            return False
    
    def generate_samples(self, num_samples: int = 16, 
                        return_intermediates: bool = False) -> torch.Tensor:
        """
        生成样本
        
        Args:
            num_samples: 样本数量
            return_intermediates: 是否返回中间结果
            
        Returns:
            生成的样本
        """
        print(f"生成 {num_samples} 个样本...")
        
        with torch.no_grad():
            samples = self.diffusion.sample(
                self.model, 
                batch_size=num_samples,
                return_intermediates=return_intermediates
            )
        
        if return_intermediates:
            x_final, intermediates = samples
            print(f"生成完成，最终样本形状: {x_final.shape}")
            print(f"中间结果数量: {len(intermediates)}")
            return x_final, intermediates
        else:
            print(f"生成完成，样本形状: {samples.shape}")
            return samples
    
    def save_samples(self, samples: torch.Tensor, save_dir: str, 
                    filename: str, save_individual: bool = True) -> str:
        """
        保存生成的样本
        
        Args:
            samples: 样本张量
            save_dir: 保存目录
            filename: 文件名
            save_individual: 是否保存单独的图像
            
        Returns:
            保存的文件路径
        """
        print(f"保存样本到: {save_dir}")
        
        save_path = save_images(
            samples, 
            save_dir, 
            filename,
            normalize=True,
            save_individual=save_individual
        )
        
        print(f"样本已保存到: {save_path}")
        return save_path
    
    def visualize_denoising(self, x_start: torch.Tensor, save_dir: str, 
                           filename: str, num_steps: int = 20) -> str:
        """
        可视化去噪过程
        
        Args:
            x_start: 原始图像
            save_dir: 保存目录
            filename: 文件名
            num_steps: 可视化步数
            
        Returns:
            保存的文件路径
        """
        print("可视化去噪过程...")
        
        denoise_path = visualize_denoising_process(
            self.model, 
            self.diffusion, 
            x_start, 
            save_dir, 
            filename,
            num_steps=num_steps
        )
        
        print(f"去噪过程可视化已保存到: {denoise_path}")
        return denoise_path
    
    def create_denoising_animation(self, intermediates: List[torch.Tensor], 
                                 save_dir: str, filename: str, fps: int = 10) -> str:
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
        print("创建去噪过程动画...")
        
        animation_path = create_denoising_animation(
            intermediates, 
            save_dir, 
            filename, 
            fps=fps
        )
        
        if animation_path:
            print(f"动画已保存到: {animation_path}")
        else:
            print("动画创建失败，请安装imageio库")
        
        return animation_path
    
    def generate_with_intermediates(self, num_samples: int = 4, 
                                  save_dir: str = None) -> None:
        """
        生成样本并保存中间结果
        
        Args:
            num_samples: 样本数量
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = self.config.system.result_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成样本和中间结果
        samples, intermediates = self.generate_samples(
            num_samples, 
            return_intermediates=True
        )
        
        # 保存最终样本
        self.save_samples(samples, save_dir, "final_samples")
        
        # 保存中间结果
        print("保存中间结果...")
        for i, intermediate in enumerate(intermediates):
            step_name = f"step_{i:03d}"
            self.save_samples(
                intermediate, 
                save_dir, 
                step_name,
                save_individual=False
            )
        
        # 创建动画
        self.create_denoising_animation(
            intermediates, 
            save_dir, 
            "denoising_process"
        )
        
        print("所有结果已保存完成！")
    
    def generate_high_quality_samples(self, num_samples: int = 16, 
                                    save_dir: str = None) -> None:
        """
        生成高质量样本
        
        Args:
            num_samples: 样本数量
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = self.config.system.result_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"生成 {num_samples} 个高质量样本...")
        
        # 生成样本
        samples = self.generate_samples(num_samples, return_intermediates=False)
        
        # 保存样本
        self.save_samples(
            samples, 
            save_dir, 
            "high_quality_samples",
            save_individual=True
        )
        
        print("高质量样本生成完成！")
    
    def interactive_sampling(self, save_dir: str = None) -> None:
        """
        交互式采样
        
        Args:
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = self.config.system.result_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n=== 交互式采样 ===")
        print("1. 生成少量样本")
        print("2. 生成大量样本")
        print("3. 可视化去噪过程")
        print("4. 退出")
        
        while True:
            try:
                choice = input("\n请选择操作 (1-4): ").strip()
                
                if choice == "1":
                    num_samples = int(input("请输入样本数量 (默认4): ") or "4")
                    self.generate_high_quality_samples(num_samples, save_dir)
                    
                elif choice == "2":
                    num_samples = int(input("请输入样本数量 (默认64): ") or "64")
                    self.generate_high_quality_samples(num_samples, save_dir)
                    
                elif choice == "3":
                    num_samples = int(input("请输入样本数量 (默认1): ") or "1")
                    samples, intermediates = self.generate_samples(
                        num_samples, 
                        return_intermediates=True
                    )
                    
                    # 可视化去噪过程
                    self.visualize_denoising(
                        samples[:1], 
                        save_dir, 
                        "interactive_denoising"
                    )
                    
                    # 创建动画
                    self.create_denoising_animation(
                        intermediates, 
                        save_dir, 
                        "interactive_animation"
                    )
                    
                elif choice == "4":
                    print("退出交互式采样")
                    break
                    
                else:
                    print("无效选择，请重新输入")
                    
            except KeyboardInterrupt:
                print("\n\n用户中断，退出")
                break
            except Exception as e:
                print(f"操作失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="扩散模型采样")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="模型检查点文件路径")
    parser.add_argument("--num_samples", type=int, default=16,
                       help="生成样本数量")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="保存目录")
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["generate", "interactive", "denoising"],
                       help="采样模式")
    parser.add_argument("--with_intermediates", action="store_true",
                       help="是否保存中间结果")
    
    args = parser.parse_args()
    
    print("=== 扩散模型采样 ===")
    
    # 获取配置
    config = get_config()
    
    # 创建采样器
    sampler = DiffusionSampler(config)
    
    # 加载模型
    if not sampler.load_model(args.checkpoint):
        print("模型加载失败，退出")
        return
    
    # 设置保存目录
    save_dir = args.save_dir or config.system.result_dir
    
    # 根据模式执行操作
    if args.mode == "generate":
        if args.with_intermediates:
            sampler.generate_with_intermediates(args.num_samples, save_dir)
        else:
            sampler.generate_high_quality_samples(args.num_samples, save_dir)
            
    elif args.mode == "interactive":
        sampler.interactive_sampling(save_dir)
        
    elif args.mode == "denoising":
        # 生成一个样本并可视化去噪过程
        samples, intermediates = sampler.generate_samples(1, return_intermediates=True)
        
        # 可视化去噪过程
        sampler.visualize_denoising(samples[:1], save_dir, "denoising_demo")
        
        # 创建动画
        sampler.create_denoising_animation(intermediates, save_dir, "denoising_demo")
    
    print("采样完成！")

if __name__ == "__main__":
    main() 
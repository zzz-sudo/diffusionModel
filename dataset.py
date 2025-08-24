"""
数据集加载模块
支持CIFAR-10数据集的下载、预处理和数据加载
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import requests
import zipfile
from tqdm import tqdm

from config import get_config

class CIFAR10Dataset:
    """CIFAR-10数据集管理类"""
    
    def __init__(self, config=None):
        """
        初始化数据集
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or get_config()
        self.data_dir = self.config.data.data_dir
        self.dataset_name = self.config.data.dataset_name
        
        # 创建数据目录
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 设置数据变换
        self.train_transform = self._get_train_transform()
        self.test_transform = self._get_test_transform()
        
        # 数据集对象
        self.train_dataset = None
        self.test_dataset = None
        
    def _get_train_transform(self):
        """获取训练数据变换"""
        transform_list = []
        
        # 数据增强
        if self.config.data.augment:
            if self.config.data.horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            transform_list.append(transforms.RandomRotation(degrees=10))
            transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        
        # 转换为张量
        transform_list.append(transforms.ToTensor())
        
        # 归一化
        if self.config.data.normalize:
            if self.config.data.normalize_range == (-1.0, 1.0):
                # 归一化到[-1, 1]
                transform_list.append(transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5]
                ))
            else:
                # 归一化到[0, 1]
                transform_list.append(transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], 
                    std=[0.2023, 0.1994, 0.2010]
                ))
        
        return transforms.Compose(transform_list)
    
    def _get_test_transform(self):
        """获取测试数据变换"""
        transform_list = [transforms.ToTensor()]
        
        # 归一化
        if self.config.data.normalize:
            if self.config.data.normalize_range == (-1.0, 1.0):
                transform_list.append(transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], 
                    std=[0.5, 0.5, 0.5]
                ))
            else:
                transform_list.append(transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], 
                    std=[0.2023, 0.1994, 0.2010]
                ))
        
        return transforms.Compose(transform_list)
    
    def download_dataset(self, force_download: bool = False):
        """
        下载CIFAR-10数据集
        
        Args:
            force_download: 是否强制重新下载
        """
        cifar_dir = os.path.join(self.data_dir, "cifar-10-batches-py")
        
        if os.path.exists(cifar_dir) and not force_download:
            print(f"CIFAR-10数据集已存在于: {cifar_dir}")
            return
        
        print("开始下载CIFAR-10数据集...")
        
        # CIFAR-10数据集URL
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        filename = "cifar-10-python.tar.gz"
        filepath = os.path.join(self.data_dir, filename)
        
        # 下载文件
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print("下载完成，开始解压...")
            
            # 解压文件
            import tarfile
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(self.data_dir)
            
            # 删除压缩文件
            os.remove(filepath)
            print(f"CIFAR-10数据集下载并解压完成: {cifar_dir}")
            
        except Exception as e:
            print(f"下载失败: {e}")
            print("尝试使用torchvision自动下载...")
            self._download_with_torchvision()
    
    def _download_with_torchvision(self):
        """使用torchvision自动下载数据集"""
        try:
            print("使用torchvision下载CIFAR-10训练集...")
            torchvision.datasets.CIFAR10(
                root=self.data_dir, 
                train=True, 
                download=True,
                transform=transforms.ToTensor()
            )
            
            print("使用torchvision下载CIFAR-10测试集...")
            torchvision.datasets.CIFAR10(
                root=self.data_dir, 
                train=False, 
                download=True,
                transform=transforms.ToTensor()
            )
            
            print("CIFAR-10数据集下载完成")
            
        except Exception as e:
            print(f"torchvision下载也失败: {e}")
            raise RuntimeError("无法下载CIFAR-10数据集")
    
    def load_dataset(self):
        """加载数据集"""
        print("加载CIFAR-10数据集...")
        
        # 下载数据集（如果不存在）
        self.download_dataset()
        
        # 加载训练集
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=False,  # 已经下载过了
            transform=self.train_transform
        )
        
        # 加载测试集
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=False,  # 已经下载过了
            transform=self.test_transform
        )
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"测试集大小: {len(self.test_dataset)}")
        
        return self.train_dataset, self.test_dataset
    
    def get_dataloaders(self, batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        获取数据加载器
        
        Args:
            batch_size: 批次大小，如果为None则使用配置中的值
            
        Returns:
            训练和测试数据加载器的元组
        """
        if self.train_dataset is None or self.test_dataset is None:
            self.load_dataset()
        
        batch_size = batch_size or self.config.training.batch_size
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            drop_last=False
        )
        
        return train_loader, test_loader
    
    def visualize_samples(self, num_samples: int = 16, save_path: Optional[str] = None):
        """
        可视化数据集样本
        
        Args:
            num_samples: 显示的样本数量
            save_path: 保存路径，如果为None则显示图像
        """
        if self.train_dataset is None:
            self.load_dataset()
        
        # 获取随机样本
        indices = torch.randperm(len(self.train_dataset))[:num_samples]
        samples = [self.train_dataset[i] for i in indices]
        
        # 创建图像网格
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, (image, label) in enumerate(samples):
            if i >= num_samples:
                break
                
            # 反归一化图像
            if self.config.data.normalize and self.config.data.normalize_range == (-1.0, 1.0):
                image = (image + 1) / 2  # 从[-1,1]转换到[0,1]
            
            # 转换为numpy数组并调整维度
            image_np = image.permute(1, 2, 0).numpy()
            image_np = np.clip(image_np, 0, 1)
            
            # 显示图像
            axes[i].imshow(image_np)
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"样本可视化已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_dataset_info(self) -> dict:
        """获取数据集信息"""
        if self.train_dataset is None:
            self.load_dataset()
        
        # 获取类别信息
        classes = self.train_dataset.classes
        
        # 统计每个类别的样本数量
        train_labels = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
        test_labels = [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
        
        train_class_counts = {}
        test_class_counts = {}
        
        for i, class_name in enumerate(classes):
            train_class_counts[class_name] = train_labels.count(i)
            test_class_counts[class_name] = test_labels.count(i)
        
        return {
            'classes': classes,
            'num_classes': len(classes),
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'train_class_counts': train_class_counts,
            'test_class_counts': test_class_counts,
            'image_size': self.config.model.image_size,
            'channels': self.config.model.in_channels
        }

class CustomDataset(Dataset):
    """自定义数据集类，支持从文件夹加载图像"""
    
    def __init__(self, root_dir: str, transform=None, image_extensions: List[str] = None):
        """
        初始化自定义数据集
        
        Args:
            root_dir: 图像文件夹路径
            transform: 图像变换
            image_extensions: 支持的图像扩展名
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp']
        
        # 获取所有图像文件
        self.image_files = []
        for ext in self.image_extensions:
            self.image_files.extend(
                [f for f in os.listdir(root_dir) if f.lower().endswith(ext)]
            )
        
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # 返回0作为默认标签

def main():
    """主函数，用于测试数据集加载"""
    print("=== 测试数据集加载 ===")
    
    # 获取配置
    config = get_config()
    
    # 创建数据集实例
    dataset_manager = CIFAR10Dataset(config)
    
    # 加载数据集
    train_dataset, test_dataset = dataset_manager.load_dataset()
    
    # 获取数据加载器
    train_loader, test_loader = dataset_manager.get_dataloaders()
    
    # 获取数据集信息
    info = dataset_manager.get_dataset_info()
    print("\n=== 数据集信息 ===")
    print(f"类别数量: {info['num_classes']}")
    print(f"类别名称: {info['classes']}")
    print(f"训练集大小: {info['train_size']}")
    print(f"测试集大小: {info['test_size']}")
    print(f"图像尺寸: {info['image_size']}x{info['image_size']}")
    print(f"通道数: {info['channels']}")
    
    # 测试数据加载
    print("\n=== 测试数据加载 ===")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"  数据形状: {data.shape}")
        print(f"  标签形状: {target.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  数据范围: [{data.min():.3f}, {data.max():.3f}]")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    # 可视化样本
    print("\n=== 可视化样本 ===")
    save_path = os.path.join(config.system.result_dir, "dataset_samples.png")
    dataset_manager.visualize_samples(num_samples=16, save_path=save_path)
    
    print("\n数据集加载测试完成！")

if __name__ == "__main__":
    main() 
"""
测试所有模块的脚本
验证扩散模型的各个组件是否正常工作
"""

import os
import sys
import torch

def test_imports():
    """测试所有模块的导入"""
    print("=== 测试模块导入 ===")
    
    try:
        from config import get_config
        print(" config 模块导入成功")
        
        from dataset import CIFAR10Dataset
        print(" dataset 模块导入成功")
        
        from model import UNetModel
        print(" model 模块导入成功")
        
        from diffusion import DiffusionProcess
        print(" diffusion 模块导入成功")
        
        from utils import setup_logging, save_checkpoint
        print(" utils 模块导入成功")
        
        from train import DiffusionTrainer
        print(" train 模块导入成功")
        
        from sample import DiffusionSampler
        print(" sample 模块导入成功")
        
        print("\n所有模块导入成功！")
        return True
        
    except Exception as e:
        print(f" 模块导入失败: {e}")
        return False

def test_config():
    """测试配置模块"""
    print("\n=== 测试配置模块 ===")
    
    try:
        from config import get_config
        
        config = get_config()
        print(f" 配置加载成功")
        print(f"  图像尺寸: {config.model.image_size}")
        print(f"  时间步数: {config.diffusion.num_timesteps}")
        print(f"  批次大小: {config.training.batch_size}")
        print(f"  学习率: {config.training.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False

def test_model():
    """测试模型模块"""
    print("\n=== 测试模型模块 ===")
    
    try:
        from config import get_config
        from model import UNetModel, count_parameters
        
        config = get_config()
        
        # 创建模型
        model = UNetModel(config)
        print(f" 模型创建成功")
        print(f"  参数数量: {count_parameters(model):,}")
        
        # 测试前向传播
        batch_size = 2
        x = torch.randn(batch_size, config.model.in_channels, 
                       config.model.image_size, config.model.image_size)
        timesteps = torch.randint(0, config.diffusion.num_timesteps, (batch_size,))
        
        # 将模型移动到CPU进行测试
        model_cpu = model.cpu()
        x_cpu = x.cpu()
        timesteps_cpu = timesteps.cpu()
        
        with torch.no_grad():
            output = model_cpu(x_cpu, timesteps_cpu)
            print(f" 前向传播成功")
            print(f"  输入形状: {x_cpu.shape}")
            print(f"  输出形状: {output.shape}")
            
            # 检查输出尺寸是否正确
            expected_shape = (batch_size, config.model.out_channels, 
                            config.model.image_size, config.model.image_size)
            if output.shape == expected_shape:
                print(f"输出尺寸正确: {expected_shape}")
            else:
                print(f"输出尺寸错误，期望: {expected_shape}，实际: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f" 模型测试失败: {e}")
        return False

def test_diffusion():
    """测试扩散过程模块"""
    print("\n=== 测试扩散过程模块 ===")
    
    try:
        from config import get_config
        from diffusion import DiffusionProcess
        
        config = get_config()
        
        # 创建扩散过程
        diffusion = DiffusionProcess(config)
        print(f" 扩散过程创建成功")
        
        # 测试噪声调度
        print(f"  β值范围: [{diffusion.noise_scheduler.beta_start:.6f}, {diffusion.noise_scheduler.beta_end:.6f}]")
        print(f"  时间步数: {diffusion.noise_scheduler.num_timesteps}")
        
        return True
        
    except Exception as e:
        print(f" 扩散过程测试失败: {e}")
        return False

def test_utils():
    """测试工具函数模块"""
    print("\n=== 测试工具函数模块 ===")
    
    try:
        from utils import set_random_seed, get_device_info
        
        # 测试随机种子设置
        set_random_seed(42)
        print(" 随机种子设置成功")
        
        # 测试设备信息获取
        device_info = get_device_info()
        print(" 设备信息获取成功")
        print(f"  CUDA可用: {device_info['cuda_available']}")
        print(f"  设备名称: {device_info['device_name']}")
        
        return True
        
    except Exception as e:
        print(f" 工具函数测试失败: {e}")
        return False

def test_dataset():
    """测试数据集模块"""
    print("\n=== 测试数据集模块 ===")
    
    try:
        from config import get_config
        from dataset import CIFAR10Dataset
        
        config = get_config()
        
        # 创建数据集管理器
        dataset_manager = CIFAR10Dataset(config)
        print(" 数据集管理器创建成功")
        
        # 获取数据集信息
        info = dataset_manager.get_dataset_info()
        print(f"  数据集信息获取成功")
        print(f"  类别数量: {info['num_classes']}")
        print(f"  训练集大小: {info['train_size']}")
        print(f"  测试集大小: {info['test_size']}")
        
        return True
        
    except Exception as e:
        print(f" 数据集测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试扩散模型的所有模块...\n")
    
    # 测试列表
    tests = [
        ("模块导入", test_imports),
        ("配置模块", test_config),
        ("模型模块", test_model),
        ("扩散过程", test_diffusion),
        ("工具函数", test_utils),
        ("数据集模块", test_dataset),
    ]
    
    # 执行测试
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"✗ {test_name} 测试失败")
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
    
    # 输出测试结果
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print(" 所有测试通过！扩散模型准备就绪。")
        print("\n下一步操作:")
        print("1. 运行 'python dataset.py' 下载数据集")
        print("2. 运行 'python train.py' 开始训练")
        print("3. 训练完成后运行 'python sample.py --checkpoint checkpoints/best_model.pth' 生成图像")
    else:
        print(" 部分测试失败，请检查错误信息。")
    
    return passed == total

if __name__ == "__main__":
    main() 

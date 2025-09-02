# 扩散模型快速开始指南

##  快速开始

### 1. 环境准备

确保你已经安装了Python 3.8+和必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 测试环境

运行测试脚本验证所有模块是否正常工作：

```bash
python test_all.py
```

如果所有测试通过，你就可以开始使用了！

### 3. 下载数据集

CIFAR-10数据集会自动下载到 `./data` 目录：

```bash
python dataset.py
```

### 4. 开始训练

开始训练扩散模型：

```bash
python train.py
```

训练过程中会：
- 自动保存检查点到 `./checkpoints` 目录
- 生成样本图像到 `./results/images` 目录
- 绘制训练曲线到 `./results/plots` 目录
- 记录训练日志到 `./results/logs` 目录

### 5. 生成图像

训练完成后，使用训练好的模型生成图像：

```bash
# 生成16个样本
python sample.py --checkpoint checkpoints/best_model.pth --num_samples 16

# 可视化去噪过程
python sample.py --checkpoint checkpoints/best_model.pth --mode denoising

# 交互式采样
python sample.py --checkpoint checkpoints/best_model.pth --mode interactive
```

##  项目结构

```
diffusionModel/
├── config.py              # 配置文件
├── dataset.py             # 数据集管理
├── model.py               # U-Net模型
├── diffusion.py           # 扩散过程
├── train.py               # 训练脚本
├── sample.py              # 采样脚本
├── utils.py               # 工具函数
├── test_all.py            # 测试脚本
├── requirements.txt       # 依赖列表
├── README.md              # 详细说明
├── QUICKSTART.md          # 快速开始指南
├── checkpoints/           # 模型检查点
├── results/               # 结果输出
│   ├── images/           # 生成图像
│   ├── plots/            # 可视化图表
│   └── logs/             # 训练日志
└── data/                  # 数据集目录
```

##  快速配置

### 修改训练参数

编辑 `config.py` 文件来调整训练参数：

```python
# 快速训练配置（用于测试）
config.training.num_epochs = 100      # 减少训练轮数
config.training.batch_size = 32       # 减少批次大小
config.diffusion.num_timesteps = 100  # 减少时间步数
```

### 使用GPU训练

确保CUDA可用，模型会自动使用GPU：

```python
config.system.device = "cuda"  # 强制使用GPU
```

##  常见问题

### Q: 训练速度太慢怎么办？
A: 
- 减少 `num_epochs` 和 `batch_size`
- 减少 `num_timesteps`
- 使用GPU训练

### Q: 内存不足怎么办？
A:
- 减少 `batch_size`
- 减少 `model_channels`
- 启用混合精度训练

### Q: 如何恢复训练？
A:
```bash
python train.py
```
程序会自动检测最新的检查点并恢复训练。

### Q: 生成的图像质量不好？
A:
- 增加训练轮数
- 调整学习率
- 检查数据预处理

##  监控训练

### 实时查看训练进度

训练过程中会显示：
- 当前轮数和损失
- 学习率变化
- 训练时间

### 查看生成结果

训练过程中会定期保存：
- 生成的样本图像
- 训练损失曲线
- 去噪过程可视化

##  下一步

1. **深入理解**: 阅读 `README.md` 了解技术细节
2. **自定义模型**: 修改 `config.py` 调整模型架构
3. **扩展功能**: 添加条件生成、风格迁移等功能
4. **性能优化**: 使用DDIM采样、噪声调度优化等

##  获取帮助

如果遇到问题：
1. 检查错误日志
2. 运行 `python test_all.py` 诊断问题
3. 查看 `README.md` 中的技术细节
4. 检查依赖是否正确安装

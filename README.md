# EDSR超分辨率重建项目

这是一个基于EDSR（Enhanced Deep Super Resolution）的超分辨率图像重建项目，实现了4倍图像放大功能。该项目包含两个版本：

1. 基础版EDSR：采用经典的残差块结构，默认包含16个残差块（可通过参数调整），每个残差块由两个3x3卷积层组成，使用ReLU激活函数。通过简单的上采样模块实现4倍放大。

2. 增强版EDSR（EAD-EDSR）：在基础版的基础上进行了多项改进：
   - 引入通道注意力机制（Channel Attention）：自适应学习特征通道的重要性
   - 引入空间注意力机制（Spatial Attention）：关注图像的空间信息
   - 增强型残差密集块：结合密集连接和注意力机制，提升特征提取能力
   - 多尺度特征提取：使用1x1、3x3、5x5、7x7四种不同尺度的卷积核提取特征
   - 渐进式上采样：采用2x2的渐进式上采样策略，每次上采样后都进行特征细化
   - 自适应损失函数：结合L1损失和SSIM损失，更好地保持图像结构信息

## 实验环境

- Python 3.10
- PyTorch 2.5.1
- Apple MPS 加速（针对 Apple Silicon Mac 优化，训练设备为Macbook）

## 依赖安装

```bash
pip install torch==2.5.1 torchvision
pip install numpy pillow matplotlib scikit-image tqdm
```

## 项目结构

```
.
├── data/               # 数据集目录
├── models/            # 模型保存目录
├── results/           # 结果输出目录
├── metrics/           # 评估指标目录
├── enhanced_edsr.py   # 增强版EDSR实现
├── edsr.py           # 基础版EDSR实现
└── edsr_bicubic_comparison.py  # EDSR&Bicubic对比实验实现
```

## 使用方法

### 1. 数据准备

将训练数据放置在 `data` 目录下，需要准备：
- 高分辨率图像（HR）
- 低分辨率图像（LR）

### 2. 模型训练

#### 基础版EDSR训练
运行基础版EDSR训练：

```bash
python edsr.py
```

#### 增强版EDSR训练
运行增强版EDSR训练（包含注意力机制和多尺度特征提取）：

```bash
python enhanced_edsr.py
```

### 3. 模型评估

运行EDSR与双三次插值的对比评估：

```bash
python edsr_bicubic_comparison.py
```

## 主要特性

- 通道注意力机制
- 空间注意力机制
- 增强型残差密集块
- 多尺度特征提取
- 渐进式上采样
- 自适应损失函数

## 性能指标

模型评估使用以下指标：
- PSNR（峰值信噪比）
- SSIM（结构相似性）

## 注意事项

1. 确保有足够的GPU内存（如果使用GPU）
2. 对于Apple Silicon Mac用户，代码会自动使用MPS加速
3. 训练过程中会自动保存最佳模型
4. 评估结果将保存在 `results` 目录下


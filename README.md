# CIFAR-10 分类器 - 三层神经网络实现

这个项目实现了一个三层神经网络分类器，用于CIFAR-10数据集的图像分类任务。项目完全使用numpy手动实现了神经网络的前向传播和反向传播算法，不依赖于任何自动微分的深度学习框架。

## 项目特点

- 使用纯numpy实现三层神经网络
- 手动实现反向传播算法
- 支持多种激活函数（ReLU、Sigmoid、Tanh）
- 实现SGD优化器和学习率调度
- 支持L2正则化
- 提供完整的超参数搜索功能
- 包含模型训练、测试和可视化

## 文件结构

```
cifar10/
├── model.py         # 神经网络模型实现
├── data.py          # 数据加载和处理
├── trainer.py       # 模型训练器和可视化
├── hyperparameter_search.py  # 超参数搜索
├── test.py          # 模型测试
├── cifar.py         # 主程序
└── README.md        # 项目说明
```

## 环境要求

- Python 3.6+
- NumPy
- Matplotlib
- pickle

## 使用方法

### 数据集下载

数据集会在首次运行时自动下载。

### 训练模型

```bash
# 使用默认参数训练模型
python cifar.py --mode train

# 自定义参数训练模型
python cifar.py --mode train --hidden_size1 200 --hidden_size2 100 --lr 0.01 --batch_size 100 --epochs 50 --reg_lambda 0.01 --activation relu
```

### 超参数搜索

```bash
# 搜索最佳学习率
python cifar.py --mode search --search_type lr

# 搜索最佳隐藏层大小
python cifar.py --mode search --search_type hidden

# 搜索最佳正则化系数
python cifar.py --mode search --search_type reg

# 搜索最佳激活函数
python cifar.py --mode search --search_type activation

# 全面搜索并使用最佳参数训练
python cifar.py --mode search --search_type all
```

### 测试模型

```bash
# 测试训练好的模型
python cifar.py --mode test --model_path ./cifar10/results/best_model.pkl
```

## 超参数说明

以下是主要的超参数及其默认值：

- `--hidden_size1`: 第一隐藏层大小（默认: 100）
- `--hidden_size2`: 第二隐藏层大小（默认: 100）
- `--activation`: 激活函数类型 [relu, sigmoid, tanh]（默认: relu）
- `--lr`: 学习率（默认: 0.01）
- `--batch_size`: 批大小（默认: 100）
- `--epochs`: 训练轮数（默认: 50）
- `--reg_lambda`: L2正则化系数（默认: 0.01）
- `--lr_decay`: 是否使用学习率衰减（默认: False）
- `--decay_rate`: 学习率衰减率（默认: 0.95）

## 实验报告

训练完成后，程序会在`./cifar10/results/`目录下生成实验报告和可视化结果。报告包括：

- 训练和验证损失/准确率曲线
- 模型权重可视化
- 混淆矩阵
- 随机样本的预测结果

## 模型架构

本项目实现的三层神经网络架构：

- 输入层：3072个神经元（32x32x3的展平图像）
- 第一隐藏层：可配置大小，默认100个神经元
- 第二隐藏层：可配置大小，默认100个神经元
- 输出层：10个神经元，对应10个类别

## 注意事项

- 第一次运行时会自动下载CIFAR-10数据集，确保网络连接正常
- 超参数搜索可能需要较长时间，建议先使用较少的epoch进行初步搜索
- 为获得最佳结果，建议使用搜索得到的最佳超参数进行完整训练

## 性能

在普通PC上使用默认超参数训练50个epoch后，该模型在CIFAR-10测试集上可以达到约45-50%的准确率。 
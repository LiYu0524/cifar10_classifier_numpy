# CIFAR-10 图像分类三层神经网络实验报告

## 1. 项目概述

本项目实现了一个三层神经网络分类器，用于CIFAR-10数据集的图像分类任务。整个实现过程完全基于numpy，手动实现了前向传播和反向传播算法，不依赖任何深度学习框架。代码和权重在[github地址](https://github.com/LiYu0524/cifar10_classifier_numpy)

### 1.1 CIFAR-10数据集

CIFAR-10是一个包含10个类别的彩色图像数据集，每个类别包含6000张32x32的彩色图像。10个类别分别是：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。数据集分为50000张训练图像和10000张测试图像。

### 1.2 模型架构

本项目实现的三层神经网络架构如下：

- 输入层：3072个神经元（32x32x3的展平图像）
- 第一隐藏层：可配置大小，默认200个神经元
- 输出层：10个神经元，对应10个类别

激活函数支持ReLU、Sigmoid和Tanh，损失函数使用交叉熵，同时支持L2正则化。

## 2. 实验结果与分析

### 2.1 超参数搜索

超参数搜索探索了以下几个关键参数的影响：

1. **学习率**：尝试了[0.001, 0.01, 0.05, 0.1, 0.5]
2. **隐藏层大小**：尝试了[(50, 50), (100, 100), (200, 100), (300, 150), (500, 200)]
3. **正则化系数**：尝试了[0.0, 0.0001, 0.001, 0.01, 0.1]
4. **激活函数**：尝试了[relu, sigmoid, tanh]

### 2.2 训练曲线

下面是使用最佳超参数训练模型的训练曲线：

![训练曲线](./cifar10/results/training_curves.png)

### 2.3 权重可视化

第一层权重可视化（重新塑形为图像）：

![第一层权重](./cifar10/results/layer1_weights_images.png)

### 2.4 测试结果

在测试集上的分类准确率：

![混淆矩阵](./cifar10/results/confusion_matrix.png)

## 3. 结论与改进方向

### 3.1 结论

本项目成功实现了一个基于numpy的三层神经网络，能够有效地对CIFAR-10数据集进行图像分类。通过手动实现反向传播算法，加深了对神经网络工作原理的理解。

### 3.2 改进方向

1. **网络深度**：增加网络层数可能提高模型表达能力
2. **优化算法**：实现更高级的优化算法，如Adam、RMSprop等
3. **卷积层**：针对图像任务，使用卷积层可能取得更好的效果
4. **数据增强**：使用数据增强技术扩充训练集
5. **批归一化**：添加批归一化层可能提高训练稳定性和收敛速度


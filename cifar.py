import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from model import TwoLayerNet
from data import load_cifar10, DataSplitter
from trainer import SGD, LearningRateScheduler, Trainer
from hyperparameter_search import HyperparameterSearch
from test import test_model

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10图像分类两层神经网络')
    
    # 数据集参数
    parser.add_argument('--dataset_dir', type=str, default='./cifar10',
                        help='数据集保存目录')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='验证集比例')
    
    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='隐藏层大小')
    parser.add_argument('--activation', type=str, default='relu',
                        help='激活函数类型 (relu, sigmoid, tanh)')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--reg_lambda', type=float, default=0.01,
                        help='L2正则化系数')
    parser.add_argument('--lr_decay', action='store_true',
                        help='是否使用学习率衰减')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                        help='学习率衰减率')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train',
                        help='运行模式 (train, search, test)')
    parser.add_argument('--search_type', type=str, default='all',
                        help='搜索类型 (lr, hidden, reg, activation, all)')
    parser.add_argument('--model_path', type=str, default='./cifar10/results/best_model.pkl',
                        help='测试模式下的模型路径')
    parser.add_argument('--model_type', type=str, default='two',
                        help='模型类型 (two 或 three)')
    
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./cifar10/results',
                        help='结果保存目录')
    parser.add_argument('--record_interval', type=int, default=1,
                        help='记录训练过程的间隔')
    
    args = parser.parse_args()
    
    # 加载和处理数据
    print("加载CIFAR-10数据集...")
    (X_train_full, y_train_full), (X_test, y_test) = load_cifar10(
        dataset_dir=args.dataset_dir,
        normalize=True,
        flatten=True,
        one_hot=False
    )
    
    # 划分训练集和验证集
    print(f"划分训练集和验证集 (验证集比例: {args.val_size})...")
    X_train, X_val, y_train, y_val = DataSplitter.train_val_split(
        X_train_full, y_train_full, val_size=args.val_size
    )
    
    print(f"数据集大小 - 训练集: {X_train.shape[0]}, 验证集: {X_val.shape[0]}, 测试集: {X_test.shape[0]}")
    
    # 根据不同模式执行相应操作
    if args.mode == 'train':
        train(args, X_train, y_train, X_val, y_val)
    elif args.mode == 'search':
        hyperparameter_search(args, X_train, y_train, X_val, y_val)
    elif args.mode == 'test':
        test(args, X_test, y_test)
    else:
        print(f"不支持的模式: {args.mode}")

def train(args, X_train, y_train, X_val, y_val):
    """训练模型"""
    print("\n开始训练模型...")
    
    # 创建模型
    model = TwoLayerNet(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation
    )
    
    # 创建优化器
    optimizer = SGD(lr=args.lr)
    
    # 学习率调度器
    lr_scheduler = None
    if args.lr_decay:
        lr_scheduler = lambda initial_lr, epoch: LearningRateScheduler.exponential_decay(
            initial_lr, epoch, decay_rate=args.decay_rate
        )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_dir=args.save_dir
    )
    
    # 训练模型
    trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        reg_lambda=args.reg_lambda,
        record_interval=args.record_interval
    )
    
    # 绘制训练结果
    trainer.plot_results()
    
    # 可视化权重
    trainer.visualize_weights(layer=1, reshape=True)
    trainer.visualize_weights(layer=2)
    
    print(f"训练完成！最佳验证准确率: {trainer.best_val_acc:.4f}")
    print(f"模型已保存至 {os.path.join(args.save_dir, 'best_model.pkl')}")

def hyperparameter_search(args, X_train, y_train, X_val, y_val):
    """超参数搜索"""
    print("\n开始超参数搜索...")
    
    # 创建超参数搜索对象
    searcher = HyperparameterSearch(save_dir=os.path.join(args.save_dir, 'hyperparam_search'))
    
    # 根据搜索类型执行相应的搜索
    if args.search_type == 'lr' or args.search_type == 'all':
        print("\n进行学习率搜索...")
        lr_list = [0.001, 0.01, 0.05, 0.1, 0.5]
        best_lr = searcher.search_learning_rate(
            X_train, y_train, X_val, y_val,
            hidden_size=args.hidden_size,
            lr_list=lr_list,
            activation=args.activation,
            batch_size=args.batch_size,
            epochs=int(args.epochs / 2),  # 减少搜索时的训练轮数
            reg_lambda=args.reg_lambda,
            record_interval=args.record_interval,
            model_type='two'
        )
        print(f"最佳学习率: {best_lr}")
        
        # 更新args中的最佳学习率
        if args.search_type == 'all':
            args.lr = best_lr
    
    if args.search_type == 'hidden' or args.search_type == 'all':
        print("\n进行隐藏层大小搜索...")
        hidden_size_list = [50, 100, 200, 300, 500]
        best_hidden_size = searcher.search_hidden_size(
            X_train, y_train, X_val, y_val,
            hidden_size_list=hidden_size_list,
            lr=args.lr,
            activation=args.activation,
            batch_size=args.batch_size,
            epochs=int(args.epochs / 2),
            reg_lambda=args.reg_lambda,
            record_interval=args.record_interval,
            model_type='two'
        )
        print(f"最佳隐藏层大小: {best_hidden_size}")
        
        # 更新args中的最佳隐藏层大小
        if args.search_type == 'all':
            args.hidden_size = best_hidden_size
    
    if args.search_type == 'reg' or args.search_type == 'all':
        print("\n进行正则化系数搜索...")
        reg_lambda_list = [0.0, 0.0001, 0.001, 0.01, 0.1]
        best_reg_lambda = searcher.search_regularization(
            X_train, y_train, X_val, y_val,
            hidden_size=args.hidden_size,
            reg_lambda_list=reg_lambda_list,
            lr=args.lr,
            activation=args.activation,
            batch_size=args.batch_size,
            epochs=int(args.epochs / 2),
            record_interval=args.record_interval,
            model_type='two'
        )
        print(f"最佳正则化系数: {best_reg_lambda}")
        
        # 更新args中的最佳正则化系数
        if args.search_type == 'all':
            args.reg_lambda = best_reg_lambda
    
    if args.search_type == 'activation' or args.search_type == 'all':
        print("\n进行激活函数搜索...")
        activation_list = ['relu', 'sigmoid', 'tanh']
        best_activation = searcher.search_activation(
            X_train, y_train, X_val, y_val,
            hidden_size=args.hidden_size,
            activation_list=activation_list,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=int(args.epochs / 2),
            reg_lambda=args.reg_lambda,
            record_interval=args.record_interval,
            model_type='two'
        )
        print(f"最佳激活函数: {best_activation}")
        
        # 更新args中的最佳激活函数
        if args.search_type == 'all':
            args.activation = best_activation
    
    # 如果是全面搜索，使用找到的最佳参数进行最终训练
    if args.search_type == 'all':
        print("\n使用最佳超参数进行最终训练...")
        print(f"最佳超参数 - 学习率: {args.lr}, 隐藏层: {args.hidden_size}, "
              f"正则化系数: {args.reg_lambda}, 激活函数: {args.activation}")
        
        # 用最佳参数进行完整训练
        train(args, X_train, y_train, X_val, y_val)
    
    # 保存搜索结果
    searcher.save_results()
    print(f"超参数搜索结果已保存至 {os.path.join(searcher.save_dir, 'search_results.pkl')}")

def test(args, X_test, y_test):
    """测试模型"""
    print("\n开始测试模型...")
    
    # 调用测试函数
    test_acc = test_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        input_size=X_test.shape[1],
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation,
        model_type=args.model_type
    )
    
    print(f"测试完成！测试集准确率: {test_acc:.4f}")

def generate_report():
    """生成实验报告"""
    report_content = """# CIFAR-10 图像分类两层神经网络实验报告

## 1. 项目概述

本项目实现了一个两层神经网络分类器，用于CIFAR-10数据集的图像分类任务。整个实现过程完全基于numpy，手动实现了前向传播和反向传播算法，不依赖任何深度学习框架。

### 1.1 CIFAR-10数据集

CIFAR-10是一个包含10个类别的彩色图像数据集，每个类别包含6000张32x32的彩色图像。10个类别分别是：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。数据集分为50000张训练图像和10000张测试图像。

### 1.2 模型架构

本项目实现的两层神经网络架构如下：

- 输入层：3072个神经元（32x32x3的展平图像）
- 隐藏层：可配置大小，默认100个神经元
- 输出层：10个神经元，对应10个类别

激活函数支持ReLU、Sigmoid和Tanh，损失函数使用交叉熵，同时支持L2正则化。

## 2. 实现细节

### 2.1 模块化设计

项目按照模块化设计原则，分为以下几个主要部分：

1. **模型部分**：实现神经网络的前向传播和反向传播算法
2. **数据处理部分**：加载和预处理CIFAR-10数据集
3. **训练部分**：实现SGD优化器、学习率调度和模型训练过程
4. **参数搜索部分**：实现超参数搜索功能
5. **测试部分**：评估模型在测试集上的性能

### 2.2 关键算法实现

#### 2.2.1 前向传播

前向传播过程计算从输入到输出的整个网络流程：

1. 第一层线性变换：Z1 = X·W1 + b1
2. 第一层激活：A1 = activation(Z1)
3. 第二层线性变换：Z2 = A1·W2 + b2
4. Softmax输出：Y = softmax(Z2)

#### 2.2.2 反向传播

反向传播算法通过链式法则计算损失函数关于网络参数的梯度：

1. 输出层误差：dZ2 = Y - T (T为one-hot编码的标签)
2. 输出层权重梯度：dW2 = A1.T·dZ2
3. 输出层偏置梯度：db2 = sum(dZ2, axis=0)
4. 隐藏层误差：dA1 = dZ2·W2.T
5. 隐藏层激活函数梯度：dZ1 = dA1 * activation_derivative(Z1)
6. 隐藏层权重梯度：dW1 = X.T·dZ1
7. 隐藏层偏置梯度：db1 = sum(dZ1, axis=0)

对于L2正则化，每个权重的梯度还需加上正则化项的梯度：λ*W。

## 3. 实验结果与分析

### 3.1 超参数搜索

超参数搜索探索了以下几个关键参数的影响：

1. **学习率**：尝试了[0.001, 0.01, 0.05, 0.1, 0.5]
2. **隐藏层大小**：尝试了[50, 100, 200, 300, 500]
3. **正则化系数**：尝试了[0.0, 0.0001, 0.001, 0.01, 0.1]
4. **激活函数**：尝试了[relu, sigmoid, tanh]

### 3.2 训练曲线

下面是使用最佳超参数训练模型的训练曲线：

![训练曲线](./cifar10/results/training_curves.png)

### 3.3 权重可视化

第一层权重可视化（重新塑形为图像）：

![第一层权重](./cifar10/results/layer1_weights_images.png)

### 3.4 测试结果

在测试集上的分类准确率：

![混淆矩阵](./cifar10/results/confusion_matrix.png)

## 4. 结论与改进方向

### 4.1 结论

本项目成功实现了一个基于numpy的两层神经网络，能够有效地对CIFAR-10数据集进行图像分类。通过手动实现反向传播算法，加深了对神经网络工作原理的理解。

### 4.2 改进方向

1. **网络深度**：增加网络层数可能提高模型表达能力
2. **优化算法**：实现更高级的优化算法，如Adam、RMSprop等
3. **卷积层**：针对图像任务，使用卷积层可能取得更好的效果
4. **数据增强**：使用数据增强技术扩充训练集
5. **批归一化**：添加批归一化层可能提高训练稳定性和收敛速度

## 5. 参考资料

1. CIFAR-10数据集：https://www.cs.toronto.edu/~kriz/cifar.html
2. 神经网络与深度学习：http://neuralnetworksanddeeplearning.com/
3. 反向传播算法：https://en.wikipedia.org/wiki/Backpropagation
"""
    
    # 保存报告
    report_dir = './cifar10/results'
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"实验报告已生成: {os.path.join(report_dir, 'report.md')}")

if __name__ == "__main__":
    main()
    generate_report()

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
import datetime
from model import ThreeLayerNet, TwoLayerNet
from trainer import SGD, LearningRateScheduler, Trainer
from data import load_cifar10, DataSplitter

class HyperparameterSearch:
    """
    超参数搜索类
    """
    def __init__(self, save_dir='./cifar10/hyperparam_results'):
        """
        初始化
        
        参数:
        - save_dir: 结果保存目录
        """
        self.save_dir = save_dir
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建日志子目录
        self.log_dir = os.path.join(save_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 存储搜索结果
        self.results = []
    
    def _create_search_log(self, search_type, params):
        """
        创建搜索日志文件
        
        参数:
        - search_type: 搜索类型
        - params: 搜索参数
        
        返回:
        - log_file: 日志文件路径
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f'{search_type}_search_{timestamp}.txt')
        
        with open(log_file, 'w') as f:
            f.write(f"{search_type.capitalize()} 搜索日志\n")
            f.write("=" * 50 + "\n")
            f.write(f"时间戳: {timestamp}\n")
            f.write(f"搜索类型: {search_type}\n")
            f.write("搜索参数:\n")
            for key, value in params.items():
                f.write(f"- {key}: {value}\n")
            f.write("=" * 50 + "\n\n")
        
        return log_file
    
    def search_learning_rate(self, X_train, y_train, X_val, y_val, 
                            hidden_size=100, hidden_size2=100,
                            lr_list=None, activation='relu',
                            batch_size=100, epochs=20, reg_lambda=0.0,
                            record_interval=5, verbose=True, model_type='three'):
        """
        搜索最佳学习率
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size: 隐藏层大小(两层网络)或第一个隐藏层大小(三层网络)
        - hidden_size2: 第二个隐藏层大小(仅用于三层网络)
        - lr_list: 学习率列表
        - activation: 激活函数
        - batch_size: 批量大小
        - epochs: 训练轮数
        - reg_lambda: L2正则化系数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        - model_type: 模型类型 ('two' 或 'three')
        
        返回:
        - 最佳学习率
        """
        if lr_list is None:
            lr_list = [0.001, 0.01, 0.05, 0.1, 0.5]
        
        # 创建日志文件
        search_params = {
            'hidden_size': hidden_size,
            'hidden_size2': hidden_size2 if model_type == 'three' else 'N/A',
            'lr_list': lr_list,
            'activation': activation,
            'batch_size': batch_size,
            'epochs': epochs,
            'reg_lambda': reg_lambda,
            'model_type': model_type
        }
        log_file = self._create_search_log('learning_rate', search_params)
        
        best_val_acc = 0
        best_lr = None
        lr_results = []
        
        for lr in lr_list:
            if verbose:
                print(f"\n开始学习率 {lr} 的测试")
            
            # 记录日志
            with open(log_file, 'a') as f:
                f.write(f"\n开始学习率 {lr} 的测试\n")
                f.write("-" * 30 + "\n")
            
            # 创建模型和训练器
            if model_type == 'two':
                model = TwoLayerNet(
                    input_size=X_train.shape[1], 
                    hidden_size=hidden_size,
                    output_size=10,
                    activation=activation
                )
            else:  # 默认为三层网络
                model = ThreeLayerNet(
                    input_size=X_train.shape[1], 
                    hidden_size1=hidden_size,
                    hidden_size2=hidden_size2,
                    output_size=10,
                    activation=activation
                )
            
            optimizer = SGD(lr=lr)
            save_dir = os.path.join(self.save_dir, f'lr_{lr}')
            
            trainer = Trainer(model, optimizer, save_dir=save_dir)
            
            # 训练模型
            train_start = time.time()
            trainer.train(X_train, y_train, X_val, y_val,
                         epochs=epochs, batch_size=batch_size,
                         reg_lambda=reg_lambda, verbose=verbose,
                         record_interval=record_interval)
            train_time = time.time() - train_start
            
            # 记录结果
            result = {
                'lr': lr,
                'hidden_size': hidden_size,
                'hidden_size2': hidden_size2 if model_type == 'three' else None,
                'activation': activation,
                'reg_lambda': reg_lambda,
                'val_acc': trainer.best_val_acc,
                'train_loss': trainer.train_loss_list,
                'val_loss': trainer.val_loss_list,
                'train_acc': trainer.train_acc_list,
                'val_acc_list': trainer.val_acc_list,
                'model_type': model_type,
                'train_time': train_time
            }
            
            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"学习率: {lr}\n")
                f.write(f"训练时间: {train_time:.2f}秒\n")
                f.write(f"最佳验证准确率: {trainer.best_val_acc:.4f}\n")
                f.write(f"最终训练损失: {trainer.train_loss_list[-1]:.4f}\n")
                f.write(f"最终验证损失: {trainer.val_loss_list[-1]:.4f}\n")
                f.write("-" * 30 + "\n")
            
            lr_results.append(result)
            self.results.append(result)
            
            # 更新最佳学习率
            if trainer.best_val_acc > best_val_acc:
                best_val_acc = trainer.best_val_acc
                best_lr = lr
                
            # 绘制训练曲线
            trainer.plot_results()
        
        # 绘制不同学习率的验证准确率比较
        self._plot_lr_comparison(lr_results)
        
        # 记录搜索结果摘要
        with open(log_file, 'a') as f:
            f.write("\n学习率搜索结果摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"最佳学习率: {best_lr}\n")
            f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
            f.write("\n所有学习率的验证准确率:\n")
            for result in lr_results:
                f.write(f"学习率 {result['lr']}: {result['val_acc']:.4f}\n")
            f.write("=" * 50 + "\n")
        
        if verbose:
            print(f"\n学习率搜索完成！")
            print(f"最佳学习率: {best_lr}, 最佳验证准确率: {best_val_acc:.4f}")
            print(f"搜索日志已保存至: {log_file}")
        
        return best_lr
    
    def search_hidden_size(self, X_train, y_train, X_val, y_val, 
                           hidden_size_list=None, hidden_size2_list=None,
                           lr=0.01, activation='relu',
                           batch_size=100, epochs=20, reg_lambda=0.0,
                           record_interval=5, verbose=True, model_type='three'):
        """
        搜索最佳隐藏层大小
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size_list: 隐藏层大小列表(两层网络)或第一个隐藏层大小列表(三层网络)
        - hidden_size2_list: 第二个隐藏层大小列表(仅用于三层网络)
        - lr: 学习率
        - activation: 激活函数
        - batch_size: 批量大小
        - epochs: 训练轮数
        - reg_lambda: L2正则化系数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        - model_type: 模型类型 ('two' 或 'three')
        
        返回:
        - 两层网络: 最佳隐藏层大小
        - 三层网络: (最佳第一隐藏层大小, 最佳第二隐藏层大小)
        """
        if hidden_size_list is None:
            hidden_size_list = [50, 100, 200, 300, 500]
        
        if hidden_size2_list is None and model_type == 'three':
            hidden_size2_list = hidden_size_list
        
        # 创建日志文件
        search_params = {
            'hidden_size_list': hidden_size_list,
            'hidden_size2_list': hidden_size2_list if model_type == 'three' else 'N/A',
            'lr': lr,
            'activation': activation,
            'batch_size': batch_size,
            'epochs': epochs,
            'reg_lambda': reg_lambda,
            'model_type': model_type
        }
        log_file = self._create_search_log('hidden_size', search_params)
        
        best_val_acc = 0
        best_hidden = None
        best_hidden2 = None
        hidden_results = []
        
        if model_type == 'two':
            # 两层网络只搜索一个隐藏层大小
            for hidden_size in hidden_size_list:
                if verbose:
                    print(f"\n开始隐藏层大小 {hidden_size} 的测试")
                
                # 记录日志
                with open(log_file, 'a') as f:
                    f.write(f"\n开始隐藏层大小 {hidden_size} 的测试\n")
                    f.write("-" * 30 + "\n")
                
                # 创建模型和训练器
                model = TwoLayerNet(
                    input_size=X_train.shape[1], 
                    hidden_size=hidden_size,
                    output_size=10,
                    activation=activation
                )
                
                optimizer = SGD(lr=lr)
                save_dir = os.path.join(self.save_dir, f'hidden_{hidden_size}')
                
                trainer = Trainer(model, optimizer, save_dir=save_dir)
                
                # 训练模型
                train_start = time.time()
                trainer.train(X_train, y_train, X_val, y_val,
                            epochs=epochs, batch_size=batch_size,
                            reg_lambda=reg_lambda, verbose=verbose,
                            record_interval=record_interval)
                train_time = time.time() - train_start
                
                # 记录结果
                result = {
                    'hidden_size': hidden_size,
                    'lr': lr,
                    'activation': activation,
                    'reg_lambda': reg_lambda,
                    'val_acc': trainer.best_val_acc,
                    'train_loss': trainer.train_loss_list,
                    'val_loss': trainer.val_loss_list,
                    'train_acc': trainer.train_acc_list,
                    'val_acc_list': trainer.val_acc_list,
                    'model_type': model_type,
                    'train_time': train_time
                }
                
                # 记录到日志
                with open(log_file, 'a') as f:
                    f.write(f"隐藏层大小: {hidden_size}\n")
                    f.write(f"训练时间: {train_time:.2f}秒\n")
                    f.write(f"最佳验证准确率: {trainer.best_val_acc:.4f}\n")
                    f.write(f"最终训练损失: {trainer.train_loss_list[-1]:.4f}\n")
                    f.write(f"最终验证损失: {trainer.val_loss_list[-1]:.4f}\n")
                    f.write("-" * 30 + "\n")
                
                hidden_results.append(result)
                self.results.append(result)
                
                # 更新最佳隐藏层大小
                if trainer.best_val_acc > best_val_acc:
                    best_val_acc = trainer.best_val_acc
                    best_hidden = hidden_size
                    
                # 绘制训练曲线
                trainer.plot_results()
                
            # 记录搜索结果摘要
            with open(log_file, 'a') as f:
                f.write("\n隐藏层大小搜索结果摘要\n")
                f.write("=" * 50 + "\n")
                f.write(f"最佳隐藏层大小: {best_hidden}\n")
                f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
                f.write("\n所有隐藏层大小的验证准确率:\n")
                for result in hidden_results:
                    f.write(f"隐藏层大小 {result['hidden_size']}: {result['val_acc']:.4f}\n")
                f.write("=" * 50 + "\n")
            
            if verbose:
                print(f"\n隐藏层大小搜索完成！")
                print(f"最佳隐藏层大小: {best_hidden}, 最佳验证准确率: {best_val_acc:.4f}")
                print(f"搜索日志已保存至: {log_file}")
            
            return best_hidden
        
        else:
            # 三层网络：搜索两个隐藏层大小的组合
            for hidden_size in hidden_size_list:
                for hidden_size2 in hidden_size2_list:
                    if verbose:
                        print(f"\n开始隐藏层大小组合 ({hidden_size}, {hidden_size2}) 的测试")
                    
                    # 记录日志
                    with open(log_file, 'a') as f:
                        f.write(f"\n开始隐藏层大小组合 ({hidden_size}, {hidden_size2}) 的测试\n")
                        f.write("-" * 30 + "\n")
                    
                    # 创建模型和训练器
                    model = ThreeLayerNet(
                        input_size=X_train.shape[1], 
                        hidden_size1=hidden_size,
                        hidden_size2=hidden_size2,
                        output_size=10,
                        activation=activation
                    )
                    
                    optimizer = SGD(lr=lr)
                    save_dir = os.path.join(self.save_dir, f'hidden_{hidden_size}_{hidden_size2}')
                    
                    trainer = Trainer(model, optimizer, save_dir=save_dir)
                    
                    # 训练模型
                    train_start = time.time()
                    trainer.train(X_train, y_train, X_val, y_val,
                                epochs=epochs, batch_size=batch_size,
                                reg_lambda=reg_lambda, verbose=verbose,
                                record_interval=record_interval)
                    train_time = time.time() - train_start
                    
                    # 记录结果
                    result = {
                        'hidden_size': hidden_size,
                        'hidden_size2': hidden_size2,
                        'lr': lr,
                        'activation': activation,
                        'reg_lambda': reg_lambda,
                        'val_acc': trainer.best_val_acc,
                        'train_loss': trainer.train_loss_list,
                        'val_loss': trainer.val_loss_list,
                        'train_acc': trainer.train_acc_list,
                        'val_acc_list': trainer.val_acc_list,
                        'model_type': model_type,
                        'train_time': train_time
                    }
                    
                    # 记录到日志
                    with open(log_file, 'a') as f:
                        f.write(f"隐藏层大小组合: ({hidden_size}, {hidden_size2})\n")
                        f.write(f"训练时间: {train_time:.2f}秒\n")
                        f.write(f"最佳验证准确率: {trainer.best_val_acc:.4f}\n")
                        f.write(f"最终训练损失: {trainer.train_loss_list[-1]:.4f}\n")
                        f.write(f"最终验证损失: {trainer.val_loss_list[-1]:.4f}\n")
                        f.write("-" * 30 + "\n")
                    
                    hidden_results.append(result)
                    self.results.append(result)
                    
                    # 更新最佳隐藏层大小
                    if trainer.best_val_acc > best_val_acc:
                        best_val_acc = trainer.best_val_acc
                        best_hidden = hidden_size
                        best_hidden2 = hidden_size2
                        
                    # 绘制训练曲线
                    trainer.plot_results()
            
            # 记录搜索结果摘要
            with open(log_file, 'a') as f:
                f.write("\n隐藏层大小搜索结果摘要\n")
                f.write("=" * 50 + "\n")
                f.write(f"最佳隐藏层大小组合: ({best_hidden}, {best_hidden2})\n")
                f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
                f.write("\n所有隐藏层大小组合的验证准确率:\n")
                for result in hidden_results:
                    f.write(f"隐藏层大小组合 ({result['hidden_size']}, {result['hidden_size2']}): {result['val_acc']:.4f}\n")
                f.write("=" * 50 + "\n")
            
            if verbose:
                print(f"\n隐藏层大小搜索完成！")
                print(f"最佳隐藏层大小组合: ({best_hidden}, {best_hidden2}), 最佳验证准确率: {best_val_acc:.4f}")
                print(f"搜索日志已保存至: {log_file}")
            
            return best_hidden, best_hidden2
    
    def search_regularization(self, X_train, y_train, X_val, y_val, 
                             hidden_size=100, hidden_size2=100,
                             reg_lambda_list=None, lr=0.01, activation='relu',
                             batch_size=100, epochs=20,
                             record_interval=5, verbose=True, model_type='three'):
        """
        搜索最佳正则化系数
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size: 隐藏层大小(两层网络)或第一个隐藏层大小(三层网络)
        - hidden_size2: 第二个隐藏层大小(仅用于三层网络)
        - reg_lambda_list: 正则化系数列表
        - lr: 学习率
        - activation: 激活函数
        - batch_size: 批量大小
        - epochs: 训练轮数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        - model_type: 模型类型 ('two' 或 'three')
        
        返回:
        - 最佳正则化系数
        """
        if reg_lambda_list is None:
            reg_lambda_list = [0.0, 0.001, 0.01, 0.1, 0.5]
            
        # 创建日志文件
        search_params = {
            'hidden_size': hidden_size,
            'hidden_size2': hidden_size2 if model_type == 'three' else 'N/A',
            'reg_lambda_list': reg_lambda_list,
            'lr': lr,
            'activation': activation,
            'batch_size': batch_size,
            'epochs': epochs,
            'model_type': model_type
        }
        log_file = self._create_search_log('regularization', search_params)
        
        best_val_acc = 0
        best_reg_lambda = None
        reg_results = []
        
        for reg_lambda in reg_lambda_list:
            if verbose:
                print(f"\n开始正则化系数 {reg_lambda} 的测试")
            
            # 记录日志
            with open(log_file, 'a') as f:
                f.write(f"\n开始正则化系数 {reg_lambda} 的测试\n")
                f.write("-" * 30 + "\n")
            
            # 创建模型和训练器
            if model_type == 'two':
                model = TwoLayerNet(
                    input_size=X_train.shape[1], 
                    hidden_size=hidden_size,
                    output_size=10,
                    activation=activation
                )
            else:  # 默认为三层网络
                model = ThreeLayerNet(
                    input_size=X_train.shape[1], 
                    hidden_size1=hidden_size,
                    hidden_size2=hidden_size2,
                    output_size=10,
                    activation=activation
                )
            
            optimizer = SGD(lr=lr)
            save_dir = os.path.join(self.save_dir, f'reg_{reg_lambda}')
            
            trainer = Trainer(model, optimizer, save_dir=save_dir)
            
            # 训练模型
            train_start = time.time()
            trainer.train(X_train, y_train, X_val, y_val,
                        epochs=epochs, batch_size=batch_size,
                        reg_lambda=reg_lambda, verbose=verbose,
                        record_interval=record_interval)
            train_time = time.time() - train_start
            
            # 记录结果
            result = {
                'reg_lambda': reg_lambda,
                'hidden_size': hidden_size,
                'hidden_size2': hidden_size2 if model_type == 'three' else None,
                'lr': lr,
                'activation': activation,
                'val_acc': trainer.best_val_acc,
                'train_loss': trainer.train_loss_list,
                'val_loss': trainer.val_loss_list,
                'train_acc': trainer.train_acc_list,
                'val_acc_list': trainer.val_acc_list,
                'model_type': model_type,
                'train_time': train_time
            }
            
            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"正则化系数: {reg_lambda}\n")
                f.write(f"训练时间: {train_time:.2f}秒\n")
                f.write(f"最佳验证准确率: {trainer.best_val_acc:.4f}\n")
                f.write(f"最终训练损失: {trainer.train_loss_list[-1]:.4f}\n")
                f.write(f"最终验证损失: {trainer.val_loss_list[-1]:.4f}\n")
                f.write("-" * 30 + "\n")
            
            reg_results.append(result)
            self.results.append(result)
            
            # 更新最佳正则化系数
            if trainer.best_val_acc > best_val_acc:
                best_val_acc = trainer.best_val_acc
                best_reg_lambda = reg_lambda
                
            # 绘制训练曲线
            trainer.plot_results()
        
        # 绘制不同正则化系数的验证准确率比较
        self._plot_reg_comparison(reg_results)
        
        # 记录搜索结果摘要
        with open(log_file, 'a') as f:
            f.write("\n正则化系数搜索结果摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"最佳正则化系数: {best_reg_lambda}\n")
            f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
            f.write("\n所有正则化系数的验证准确率:\n")
            for result in reg_results:
                f.write(f"正则化系数 {result['reg_lambda']}: {result['val_acc']:.4f}\n")
            f.write("=" * 50 + "\n")
        
        if verbose:
            print(f"\n正则化系数搜索完成！")
            print(f"最佳正则化系数: {best_reg_lambda}, 最佳验证准确率: {best_val_acc:.4f}")
            print(f"搜索日志已保存至: {log_file}")
        
        return best_reg_lambda
    
    def search_activation(self, X_train, y_train, X_val, y_val, 
                         hidden_size=100, hidden_size2=100,
                         activation_list=None, lr=0.01, 
                         batch_size=100, epochs=20, reg_lambda=0.0,
                         record_interval=5, verbose=True, model_type='three'):
        """
        搜索最佳激活函数
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size: 隐藏层大小(两层网络)或第一个隐藏层大小(三层网络)
        - hidden_size2: 第二个隐藏层大小(仅用于三层网络)
        - activation_list: 激活函数列表
        - lr: 学习率
        - batch_size: 批量大小
        - epochs: 训练轮数
        - reg_lambda: L2正则化系数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        - model_type: 模型类型 ('two' 或 'three')
        
        返回:
        - 最佳激活函数
        """
        if activation_list is None:
            activation_list = ['relu', 'sigmoid', 'tanh']
            
        # 创建日志文件
        search_params = {
            'hidden_size': hidden_size,
            'hidden_size2': hidden_size2 if model_type == 'three' else 'N/A',
            'activation_list': activation_list,
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'reg_lambda': reg_lambda,
            'model_type': model_type
        }
        log_file = self._create_search_log('activation', search_params)
        
        best_val_acc = 0
        best_activation = None
        activation_results = []
        
        for activation in activation_list:
            if verbose:
                print(f"\n开始激活函数 {activation} 的测试")
            
            # 记录日志
            with open(log_file, 'a') as f:
                f.write(f"\n开始激活函数 {activation} 的测试\n")
                f.write("-" * 30 + "\n")
            
            # 创建模型和训练器
            if model_type == 'two':
                model = TwoLayerNet(
                    input_size=X_train.shape[1], 
                    hidden_size=hidden_size,
                    output_size=10,
                    activation=activation
                )
            else:  # 默认为三层网络
                model = ThreeLayerNet(
                    input_size=X_train.shape[1], 
                    hidden_size1=hidden_size,
                    hidden_size2=hidden_size2,
                    output_size=10,
                    activation=activation
                )
            
            optimizer = SGD(lr=lr)
            save_dir = os.path.join(self.save_dir, f'activation_{activation}')
            
            trainer = Trainer(model, optimizer, save_dir=save_dir)
            
            # 训练模型
            train_start = time.time()
            trainer.train(X_train, y_train, X_val, y_val,
                        epochs=epochs, batch_size=batch_size,
                        reg_lambda=reg_lambda, verbose=verbose,
                        record_interval=record_interval)
            train_time = time.time() - train_start
            
            # 记录结果
            result = {
                'activation': activation,
                'hidden_size': hidden_size,
                'hidden_size2': hidden_size2 if model_type == 'three' else None,
                'lr': lr,
                'reg_lambda': reg_lambda,
                'val_acc': trainer.best_val_acc,
                'train_loss': trainer.train_loss_list,
                'val_loss': trainer.val_loss_list,
                'train_acc': trainer.train_acc_list,
                'val_acc_list': trainer.val_acc_list,
                'model_type': model_type,
                'train_time': train_time
            }
            
            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"激活函数: {activation}\n")
                f.write(f"训练时间: {train_time:.2f}秒\n")
                f.write(f"最佳验证准确率: {trainer.best_val_acc:.4f}\n")
                f.write(f"最终训练损失: {trainer.train_loss_list[-1]:.4f}\n")
                f.write(f"最终验证损失: {trainer.val_loss_list[-1]:.4f}\n")
                f.write("-" * 30 + "\n")
            
            activation_results.append(result)
            self.results.append(result)
            
            # 更新最佳激活函数
            if trainer.best_val_acc > best_val_acc:
                best_val_acc = trainer.best_val_acc
                best_activation = activation
                
            # 绘制训练曲线
            trainer.plot_results()
        
        # 绘制不同激活函数的验证准确率比较
        self._plot_activation_comparison(activation_results)
        
        # 记录搜索结果摘要
        with open(log_file, 'a') as f:
            f.write("\n激活函数搜索结果摘要\n")
            f.write("=" * 50 + "\n")
            f.write(f"最佳激活函数: {best_activation}\n")
            f.write(f"最佳验证准确率: {best_val_acc:.4f}\n")
            f.write("\n所有激活函数的验证准确率:\n")
            for result in activation_results:
                f.write(f"激活函数 {result['activation']}: {result['val_acc']:.4f}\n")
            f.write("=" * 50 + "\n")
        
        if verbose:
            print(f"\n激活函数搜索完成！")
            print(f"最佳激活函数: {best_activation}, 最佳验证准确率: {best_val_acc:.4f}")
            print(f"搜索日志已保存至: {log_file}")
        
        return best_activation
    
    def _plot_lr_comparison(self, results):
        """绘制不同学习率的比较"""
        plt.figure(figsize=(12, 5))
        
        # 验证准确率比较
        plt.subplot(1, 2, 1)
        for result in results:
            epochs = range(1, len(result['val_acc_list']) + 1)
            plt.plot(epochs, result['val_acc_list'], label=f"lr={result['lr']}")
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.title('Learning Rate Comparison - Validation Accuracy')
        plt.legend()
        
        # 最终验证准确率比较
        plt.subplot(1, 2, 2)
        lr_values = [result['lr'] for result in results]
        val_accs = [result['val_acc'] for result in results]
        
        plt.bar(range(len(lr_values)), val_accs)
        plt.xticks(range(len(lr_values)), [str(lr) for lr in lr_values])
        plt.xlabel('Learning Rate')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Learning Rate vs. Best Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'lr_comparison.png'))
        plt.close()
    
    def _plot_hidden_size_comparison(self, results):
        """绘制不同隐藏层大小的比较"""
        plt.figure(figsize=(12, 5))
        
        # 验证准确率比较
        plt.subplot(1, 2, 1)
        for result in results:
            epochs = range(1, len(result['val_acc_list']) + 1)
            if result['hidden_size2'] is not None:
                label = f"hidden=({result['hidden_size']}, {result['hidden_size2']})"
            else:
                label = f"hidden={result['hidden_size']}"
            plt.plot(epochs, result['val_acc_list'], label=label)
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.title('Hidden Size Comparison - Validation Accuracy')
        plt.legend()
        
        # 最终验证准确率比较
        plt.subplot(1, 2, 2)
        if results[0]['hidden_size2'] is not None:
            hidden_labels = [f"({r['hidden_size']}, {r['hidden_size2']})" for r in results]
        else:
            hidden_labels = [str(r['hidden_size']) for r in results]
        val_accs = [result['val_acc'] for result in results]
        
        plt.bar(range(len(hidden_labels)), val_accs)
        plt.xticks(range(len(hidden_labels)), hidden_labels, rotation=45 if len(hidden_labels[0]) > 3 else 0)
        plt.xlabel('Hidden Size')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Hidden Size vs. Best Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'hidden_size_comparison.png'))
        plt.close()
    
    def _plot_reg_comparison(self, results):
        """绘制不同正则化系数的比较"""
        plt.figure(figsize=(12, 5))
        
        # 验证准确率比较
        plt.subplot(1, 2, 1)
        for result in results:
            epochs = range(1, len(result['val_acc_list']) + 1)
            plt.plot(epochs, result['val_acc_list'], label=f"reg={result['reg_lambda']}")
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.title('Regularization Comparison - Validation Accuracy')
        plt.legend()
        
        # 最终验证准确率比较
        plt.subplot(1, 2, 2)
        reg_values = [result['reg_lambda'] for result in results]
        val_accs = [result['val_acc'] for result in results]
        
        plt.bar(range(len(reg_values)), val_accs)
        plt.xticks(range(len(reg_values)), [str(reg) for reg in reg_values])
        plt.xlabel('Regularization Lambda')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Regularization vs. Best Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'reg_comparison.png'))
        plt.close()
    
    def _plot_activation_comparison(self, results):
        """绘制不同激活函数的比较"""
        plt.figure(figsize=(12, 5))
        
        # 验证准确率比较
        plt.subplot(1, 2, 1)
        for result in results:
            epochs = range(1, len(result['val_acc_list']) + 1)
            plt.plot(epochs, result['val_acc_list'], label=f"act={result['activation']}")
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Accuracy')
        plt.title('Activation Function Comparison - Validation Accuracy')
        plt.legend()
        
        # 最终验证准确率比较
        plt.subplot(1, 2, 2)
        act_values = [result['activation'] for result in results]
        val_accs = [result['val_acc'] for result in results]
        
        plt.bar(range(len(act_values)), val_accs)
        plt.xticks(range(len(act_values)), act_values)
        plt.xlabel('Activation Function')
        plt.ylabel('Best Validation Accuracy')
        plt.title('Activation Function vs. Best Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'activation_comparison.png'))
        plt.close()
    
    def save_results(self, file_name='search_results.pkl'):
        """保存搜索结果"""
        with open(os.path.join(self.save_dir, file_name), 'wb') as f:
            pickle.dump(self.results, f)
    
    def load_results(self, file_name='search_results.pkl'):
        """加载搜索结果"""
        with open(os.path.join(self.save_dir, file_name), 'rb') as f:
            self.results = pickle.load(f)
    
    def grid_search(self, X_train, y_train, X_val, y_val,
                   lr_list=None, hidden_size_list=None, 
                   reg_lambda_list=None, activation_list=None,
                   batch_size=100, epochs=10, record_interval=5,
                   verbose=True):
        """
        网格搜索最佳超参数组合
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - lr_list: 学习率列表
        - hidden_size_list: 隐藏层大小列表
        - reg_lambda_list: 正则化系数列表
        - activation_list: 激活函数列表
        - batch_size: 批量大小
        - epochs: 训练轮数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        
        返回:
        - 最佳超参数组合和验证准确率
        """
        # 设置默认超参数列表
        if lr_list is None:
            lr_list = [0.01, 0.1]
        if hidden_size_list is None:
            hidden_size_list = [(100, 100), (200, 100)]
        if reg_lambda_list is None:
            reg_lambda_list = [0.0, 0.01]
        if activation_list is None:
            activation_list = ['relu', 'tanh']
        
        best_val_acc = 0
        best_params = None
        
        # 记录搜索开始时间
        start_time = time.time()
        
        # 计算总组合数
        total_combinations = len(lr_list) * len(hidden_size_list) * len(reg_lambda_list) * len(activation_list)
        current_combination = 0
        
        for lr in lr_list:
            for hidden_size in hidden_size_list:
                hidden_size1, hidden_size2 = hidden_size
                for reg_lambda in reg_lambda_list:
                    for activation in activation_list:
                        current_combination += 1
                        
                        if verbose:
                            print(f"\n[{current_combination}/{total_combinations}] 测试组合: lr={lr}, "
                                 f"hidden=({hidden_size1}, {hidden_size2}), "
                                 f"reg={reg_lambda}, activation={activation}")
                        
                        # 创建模型和训练器
                        model = ThreeLayerNet(input_size=X_train.shape[1], 
                                             hidden_size1=hidden_size1,
                                             hidden_size2=hidden_size2,
                                             output_size=10,
                                             activation=activation)
                        
                        optimizer = SGD(lr=lr)
                        save_dir = os.path.join(self.save_dir, 
                                              f'grid_lr{lr}_h{hidden_size1}_{hidden_size2}_reg{reg_lambda}_act{activation}')
                        
                        trainer = Trainer(model, optimizer, save_dir=save_dir)
                        
                        # 训练模型
                        trainer.train(X_train, y_train, X_val, y_val,
                                     epochs=epochs, batch_size=batch_size,
                                     reg_lambda=reg_lambda, verbose=verbose,
                                     record_interval=record_interval)
                        
                        # 记录结果
                        result = {
                            'lr': lr,
                            'hidden_size1': hidden_size1,
                            'hidden_size2': hidden_size2,
                            'activation': activation,
                            'reg_lambda': reg_lambda,
                            'val_acc': trainer.best_val_acc,
                            'train_loss': trainer.train_loss_list,
                            'val_loss': trainer.val_loss_list,
                            'train_acc': trainer.train_acc_list,
                            'val_acc_list': trainer.val_acc_list
                        }
                        
                        self.results.append(result)
                        
                        # 更新最佳超参数
                        if trainer.best_val_acc > best_val_acc:
                            best_val_acc = trainer.best_val_acc
                            best_params = {
                                'lr': lr,
                                'hidden_size1': hidden_size1,
                                'hidden_size2': hidden_size2,
                                'activation': activation,
                                'reg_lambda': reg_lambda,
                                'val_acc': trainer.best_val_acc
                            }
        
        # 计算总搜索时间
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n网格搜索完成！总时间: {total_time:.2f}秒")
            print(f"最佳参数组合: {best_params}")
            print(f"最佳验证准确率: {best_val_acc:.4f}")
        
        # 保存搜索结果
        self.save_results()
        
        return best_params 
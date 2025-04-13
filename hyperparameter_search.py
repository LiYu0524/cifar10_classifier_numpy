import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from model import ThreeLayerNet
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
        
        # 存储搜索结果
        self.results = []
    
    def search_learning_rate(self, X_train, y_train, X_val, y_val, 
                            hidden_size1=100, hidden_size2=100,
                            lr_list=None, activation='relu',
                            batch_size=100, epochs=20, reg_lambda=0.0,
                            record_interval=5, verbose=True):
        """
        搜索最佳学习率
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size1: 第一个隐藏层大小
        - hidden_size2: 第二个隐藏层大小
        - lr_list: 学习率列表
        - activation: 激活函数
        - batch_size: 批量大小
        - epochs: 训练轮数
        - reg_lambda: L2正则化系数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        
        返回:
        - 最佳学习率
        """
        if lr_list is None:
            lr_list = [0.001, 0.01, 0.05, 0.1, 0.5]
        
        best_val_acc = 0
        best_lr = None
        lr_results = []
        
        for lr in lr_list:
            if verbose:
                print(f"\n开始学习率 {lr} 的测试")
            
            # 创建模型和训练器
            model = ThreeLayerNet(input_size=X_train.shape[1], 
                                 hidden_size1=hidden_size1,
                                 hidden_size2=hidden_size2,
                                 output_size=10,
                                 activation=activation)
            
            optimizer = SGD(lr=lr)
            save_dir = os.path.join(self.save_dir, f'lr_{lr}')
            
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
        
        if verbose:
            print(f"\n学习率搜索完成！")
            print(f"最佳学习率: {best_lr}, 最佳验证准确率: {best_val_acc:.4f}")
        
        return best_lr
    
    def search_hidden_size(self, X_train, y_train, X_val, y_val,
                           hidden_size_list=None, lr=0.01, activation='relu',
                           batch_size=100, epochs=20, reg_lambda=0.0,
                           record_interval=5, verbose=True):
        """
        搜索最佳隐藏层大小
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size_list: 隐藏层大小列表，每个元素是一个包含两个隐藏层大小的元组
        - lr: 学习率
        - activation: 激活函数
        - batch_size: 批量大小
        - epochs: 训练轮数
        - reg_lambda: L2正则化系数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        
        返回:
        - 最佳隐藏层大小 (hidden_size1, hidden_size2)
        """
        if hidden_size_list is None:
            hidden_size_list = [(50, 50), (100, 100), (200, 100), (300, 150), (500, 200)]
        
        best_val_acc = 0
        best_hidden_size = None
        hidden_size_results = []
        
        for hidden_size1, hidden_size2 in hidden_size_list:
            if verbose:
                print(f"\n开始隐藏层大小 ({hidden_size1}, {hidden_size2}) 的测试")
            
            # 创建模型和训练器
            model = ThreeLayerNet(input_size=X_train.shape[1], 
                                 hidden_size1=hidden_size1,
                                 hidden_size2=hidden_size2,
                                 output_size=10,
                                 activation=activation)
            
            optimizer = SGD(lr=lr)
            save_dir = os.path.join(self.save_dir, f'hidden_{hidden_size1}_{hidden_size2}')
            
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
            
            hidden_size_results.append(result)
            self.results.append(result)
            
            # 更新最佳隐藏层大小
            if trainer.best_val_acc > best_val_acc:
                best_val_acc = trainer.best_val_acc
                best_hidden_size = (hidden_size1, hidden_size2)
                
            # 绘制训练曲线
            trainer.plot_results()
        
        # 绘制不同隐藏层大小的验证准确率比较
        self._plot_hidden_size_comparison(hidden_size_results)
        
        if verbose:
            print(f"\n隐藏层大小搜索完成！")
            print(f"最佳隐藏层大小: {best_hidden_size}, 最佳验证准确率: {best_val_acc:.4f}")
        
        return best_hidden_size
    
    def search_regularization(self, X_train, y_train, X_val, y_val,
                              hidden_size1=100, hidden_size2=100,
                              reg_lambda_list=None, lr=0.01, activation='relu',
                              batch_size=100, epochs=20,
                              record_interval=5, verbose=True):
        """
        搜索最佳正则化系数
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size1: 第一个隐藏层大小
        - hidden_size2: 第二个隐藏层大小
        - reg_lambda_list: 正则化系数列表
        - lr: 学习率
        - activation: 激活函数
        - batch_size: 批量大小
        - epochs: 训练轮数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        
        返回:
        - 最佳正则化系数
        """
        if reg_lambda_list is None:
            reg_lambda_list = [0.0, 0.001, 0.01, 0.1, 0.5]
        
        best_val_acc = 0
        best_reg_lambda = None
        reg_results = []
        
        for reg_lambda in reg_lambda_list:
            if verbose:
                print(f"\n开始正则化系数 {reg_lambda} 的测试")
            
            # 创建模型和训练器
            model = ThreeLayerNet(input_size=X_train.shape[1], 
                                 hidden_size1=hidden_size1,
                                 hidden_size2=hidden_size2,
                                 output_size=10,
                                 activation=activation)
            
            optimizer = SGD(lr=lr)
            save_dir = os.path.join(self.save_dir, f'reg_{reg_lambda}')
            
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
        
        if verbose:
            print(f"\n正则化系数搜索完成！")
            print(f"最佳正则化系数: {best_reg_lambda}, 最佳验证准确率: {best_val_acc:.4f}")
        
        return best_reg_lambda
    
    def search_activation(self, X_train, y_train, X_val, y_val,
                          hidden_size1=100, hidden_size2=100,
                          activation_list=None, lr=0.01,
                          batch_size=100, epochs=20, reg_lambda=0.0,
                          record_interval=5, verbose=True):
        """
        搜索最佳激活函数
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - hidden_size1: 第一个隐藏层大小
        - hidden_size2: 第二个隐藏层大小
        - activation_list: 激活函数列表
        - lr: 学习率
        - batch_size: 批量大小
        - epochs: 训练轮数
        - reg_lambda: L2正则化系数
        - record_interval: 记录结果的间隔
        - verbose: 是否打印过程
        
        返回:
        - 最佳激活函数
        """
        if activation_list is None:
            activation_list = ['relu', 'sigmoid', 'tanh']
        
        best_val_acc = 0
        best_activation = None
        activation_results = []
        
        for activation in activation_list:
            if verbose:
                print(f"\n开始激活函数 {activation} 的测试")
            
            # 创建模型和训练器
            model = ThreeLayerNet(input_size=X_train.shape[1], 
                                 hidden_size1=hidden_size1,
                                 hidden_size2=hidden_size2,
                                 output_size=10,
                                 activation=activation)
            
            optimizer = SGD(lr=lr)
            save_dir = os.path.join(self.save_dir, f'activation_{activation}')
            
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
        
        if verbose:
            print(f"\n激活函数搜索完成！")
            print(f"最佳激活函数: {best_activation}, 最佳验证准确率: {best_val_acc:.4f}")
        
        return best_activation
    
    def _plot_lr_comparison(self, results):
        """绘制不同学习率的比较结果"""
        plt.figure(figsize=(15, 5))
        
        # 绘制验证准确率对比
        plt.subplot(1, 2, 1)
        lr_list = [result['lr'] for result in results]
        val_acc_list = [result['val_acc'] for result in results]
        
        plt.bar(range(len(lr_list)), val_acc_list)
        plt.xticks(range(len(lr_list)), [str(lr) for lr in lr_list])
        plt.xlabel('Learning Rate')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy Comparison of Different Learning Rates')
        
        # 绘制验证损失曲线
        plt.subplot(1, 2, 2)
        epochs = range(1, len(results[0]['val_loss']) + 1)
        
        for result in results:
            plt.plot(epochs, result['val_loss'], label=f"lr={result['lr']}")
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Curves of Different Learning Rates')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'lr_comparison.png'))
        plt.close()
    
    def _plot_hidden_size_comparison(self, results):
        """绘制不同隐藏层大小的比较结果"""
        plt.figure(figsize=(15, 5))
        
        # 绘制验证准确率对比
        plt.subplot(1, 2, 1)
        hidden_size_list = [f"({result['hidden_size1']}, {result['hidden_size2']})" for result in results]
        val_acc_list = [result['val_acc'] for result in results]
        
        plt.bar(range(len(hidden_size_list)), val_acc_list)
        plt.xticks(range(len(hidden_size_list)), hidden_size_list, rotation=45)
        plt.xlabel('Hidden Layer Sizes (layer1, layer2)')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy Comparison of Different Hidden Layer Sizes')
        
        # 绘制验证损失曲线
        plt.subplot(1, 2, 2)
        epochs = range(1, len(results[0]['val_loss']) + 1)
        
        for result in results:
            plt.plot(epochs, result['val_loss'], 
                    label=f"hidden=({result['hidden_size1']}, {result['hidden_size2']})")
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Curves of Different Hidden Layer Sizes')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'hidden_size_comparison.png'))
        plt.close()
    
    def _plot_reg_comparison(self, results):
        """绘制不同正则化系数的比较结果"""
        plt.figure(figsize=(15, 5))
        
        # 绘制验证准确率对比
        plt.subplot(1, 2, 1)
        reg_lambda_list = [result['reg_lambda'] for result in results]
        val_acc_list = [result['val_acc'] for result in results]
        
        plt.bar(range(len(reg_lambda_list)), val_acc_list)
        plt.xticks(range(len(reg_lambda_list)), [str(reg) for reg in reg_lambda_list])
        plt.xlabel('Regularization Coefficient')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy Comparison of Different Regularization Coefficients')
        
        # 绘制验证损失曲线
        plt.subplot(1, 2, 2)
        epochs = range(1, len(results[0]['val_loss']) + 1)
        
        for result in results:
            plt.plot(epochs, result['val_loss'], label=f"reg={result['reg_lambda']}")
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Curves of Different Regularization Coefficients')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'reg_comparison.png'))
        plt.close()
    
    def _plot_activation_comparison(self, results):
        """绘制不同激活函数的比较结果"""
        plt.figure(figsize=(15, 5))
        
        # 绘制验证准确率对比
        plt.subplot(1, 2, 1)
        activation_list = [result['activation'] for result in results]
        val_acc_list = [result['val_acc'] for result in results]
        
        plt.bar(range(len(activation_list)), val_acc_list)
        plt.xticks(range(len(activation_list)), activation_list)
        plt.xlabel('Activation Function')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy Comparison of Different Activation Functions')
        
        # 绘制验证损失曲线
        plt.subplot(1, 2, 2)
        epochs = range(1, len(results[0]['val_loss']) + 1)
        
        for result in results:
            plt.plot(epochs, result['val_loss'], label=f"{result['activation']}")
        
        plt.xlabel('Epochs')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss Curves of Different Activation Functions')
        plt.legend()
        
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
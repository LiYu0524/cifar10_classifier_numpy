import numpy as np
import os
import time
import matplotlib.pyplot as plt

class SGD:
    """
    随机梯度下降优化器
    """
    def __init__(self, lr=0.01):
        """
        初始化
        
        参数:
        - lr: 学习率
        """
        self.lr = lr
    
    def update(self, params, grads):
        """
        更新参数
        
        参数:
        - params: 模型参数字典
        - grads: 参数梯度字典
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class LearningRateScheduler:
    """
    学习率调度器
    """
    @staticmethod
    def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
        """
        学习率阶梯衰减
        
        参数:
        - initial_lr: 初始学习率
        - epoch: 当前epoch
        - drop_rate: 学习率衰减率
        - epochs_drop: 每隔多少个epoch衰减一次
        
        返回:
        - 新的学习率
        """
        factor = np.power(drop_rate, np.floor((1 + epoch) / epochs_drop))
        return initial_lr * factor
    
    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """
        指数衰减学习率
        
        参数:
        - initial_lr: 初始学习率
        - epoch: 当前epoch
        - decay_rate: 衰减率
        
        返回:
        - 新的学习率
        """
        return initial_lr * (decay_rate ** epoch)

class Trainer:
    """
    神经网络训练器
    """
    def __init__(self, model, optimizer, lr_scheduler=None, save_dir='./cifar10/results'):
        """
        初始化
        
        参数:
        - model: 神经网络模型
        - optimizer: 优化器
        - lr_scheduler: 学习率调度器函数
        - save_dir: 结果保存目录
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_dir = save_dir
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 记录训练过程
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_acc_list = []
        self.val_acc_list = []
        self.best_val_acc = 0
        self.initial_lr = optimizer.lr
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=100, reg_lambda=0.0,
              verbose=True, record_interval=1):
        """
        训练模型
        
        参数:
        - X_train: 训练数据
        - y_train: 训练标签
        - X_val: 验证数据
        - y_val: 验证标签
        - epochs: 训练轮数
        - batch_size: 批量大小
        - reg_lambda: L2正则化系数
        - verbose: 是否打印训练过程
        - record_interval: 记录结果的间隔
        """
        # 计算batch数量
        train_size = X_train.shape[0]
        iter_per_epoch = max(train_size // batch_size, 1)
        max_iter = epochs * iter_per_epoch
        
        # 记录总训练时间
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # 更新学习率
            if self.lr_scheduler:
                self.optimizer.lr = self.lr_scheduler(self.initial_lr, epoch)
            
            # 记录当前epoch的开始时间
            epoch_start = time.time()
            
            # 打乱训练数据
            indices = np.arange(train_size)
            np.random.shuffle(indices)
            X_batch_shuffle = X_train[indices]
            y_batch_shuffle = y_train[indices]
            
            # 按批次训练
            epoch_loss = 0
            for i in range(iter_per_epoch):
                batch_start = i * batch_size
                batch_end = min(batch_start + batch_size, train_size)
                
                # 获取当前批次数据
                X_batch = X_batch_shuffle[batch_start:batch_end]
                y_batch = y_batch_shuffle[batch_start:batch_end]
                
                # 前向传播
                loss = self._update(X_batch, y_batch, reg_lambda)
                epoch_loss += loss
            
            # 计算平均损失
            epoch_loss /= iter_per_epoch
            
            # 每隔record_interval个epoch记录一次训练和验证指标
            if epoch % record_interval == 0:
                # 计算训练集损失和准确率
                train_loss = self.model.loss(X_train, y_train, reg_lambda)
                train_acc = self.model.accuracy(X_train, y_train)
                self.train_loss_list.append(train_loss)
                self.train_acc_list.append(train_acc)
                
                # 计算验证集损失和准确率
                val_loss = self.model.loss(X_val, y_val, reg_lambda)
                val_acc = self.model.accuracy(X_val, y_val)
                self.val_loss_list.append(val_loss)
                self.val_acc_list.append(val_acc)
                
                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.model.save_params(os.path.join(self.save_dir, 'best_model.pkl'))
                
                # 计算当前epoch的训练时间
                epoch_time = time.time() - epoch_start
                
                if verbose:
                    print(f"Epoch {epoch}/{epochs}, 时间: {epoch_time:.2f}秒, 学习率: {self.optimizer.lr:.6f}")
                    print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
                    print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
                    print("-" * 50)
        
        # 训练完成后保存最终模型
        self.model.save_params(os.path.join(self.save_dir, 'final_model.pkl'))
        
        # 计算总训练时间
        total_time = time.time() - start_time
        if verbose:
            print(f"训练完成！总时间: {total_time:.2f}秒")
            print(f"最佳验证准确率: {self.best_val_acc:.4f}")
    
    def _update(self, X_batch, y_batch, reg_lambda):
        """
        单步更新
        
        参数:
        - X_batch: 输入数据批次
        - y_batch: 标签批次
        - reg_lambda: 正则化系数
        
        返回:
        - 损失值
        """
        # 前向传播
        self.model.forward(X_batch)
        
        # 计算损失
        loss = self.model.loss_function.forward(self.model.z3, y_batch)
        
        # L2正则化
        if reg_lambda > 0:
            W1, W2, W3 = self.model.params['W1'], self.model.params['W2'], self.model.params['W3']
            reg_loss = 0.5 * reg_lambda * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
            loss += reg_loss
        
        # 反向传播
        grads = self.model.backward(y_batch, reg_lambda)
        
        # 使用优化器更新参数
        self.optimizer.update(self.model.params, grads)
        
        return loss
    
    def plot_results(self, save_dir=None):
        """
        绘制训练结果
        
        参数:
        - save_dir: 图像保存目录，如果为None则使用默认目录
        """
        if save_dir is None:
            save_dir = self.save_dir
        
        # 确保目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建记录的epoch索引
        epochs = range(1, len(self.train_loss_list) + 1)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_list, label='Training Loss')
        plt.plot(epochs, self.val_loss_list, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_acc_list, label='Training Accuracy')
        plt.plot(epochs, self.val_acc_list, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
    
    def visualize_weights(self, save_dir=None, layer=1, reshape=True):
        """
        可视化模型权重
        
        参数:
        - save_dir: 图像保存目录，如果为None则使用默认目录
        - layer: 可视化哪一层的权重 (1, 2, 或 3)
        - reshape: 是否将权重重新塑形为图像（仅对第一层有效）
        """
        if save_dir is None:
            save_dir = self.save_dir
        
        # 确保目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 获取权重
        if layer == 1:
            weights = self.model.params['W1']
            title = 'First Layer Weights'
        elif layer == 2:
            weights = self.model.params['W2']
            title = 'Second Layer Weights'
        elif layer == 3:
            weights = self.model.params['W3']
            title = 'Third Layer Weights'
        else:
            raise ValueError("Layer index must be 1, 2 or 3")
        
        # 对于第一层，可以将权重重新塑形为图像
        if layer == 1 and reshape:
            # 假设输入是展平的32x32x3的图像
            n_weights = min(100, weights.shape[1])  # 最多可视化100个神经元
            
            # 计算合适的图像排列
            grid_size = int(np.ceil(np.sqrt(n_weights)))
            
            plt.figure(figsize=(15, 15))
            for i in range(n_weights):
                if i >= grid_size * grid_size:
                    break
                    
                plt.subplot(grid_size, grid_size, i+1)
                
                # 重新塑形权重为图像
                img = weights[:, i].reshape(32, 32, 3)
                
                # 标准化权重以便可视化
                img = (img - img.min()) / (img.max() - img.min())
                
                plt.imshow(img)
                plt.axis('off')
            
            plt.suptitle(f'{title} (Reshaped as Images)', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.savefig(os.path.join(save_dir, f'layer{layer}_weights_images.png'))
            plt.close()
        
        # 使用热力图可视化权重矩阵
        plt.figure(figsize=(12, 10))
        
        # 对于大权重矩阵，随机采样一部分可视化
        if weights.size > 1000000:  # 超过100万个元素
            # 随机选择行和列
            max_elements = 1000
            rows = min(weights.shape[0], int(np.sqrt(max_elements)))
            cols = min(weights.shape[1], int(np.sqrt(max_elements)))
            
            row_indices = np.random.choice(weights.shape[0], rows, replace=False)
            col_indices = np.random.choice(weights.shape[1], cols, replace=False)
            
            sampled_weights = weights[np.ix_(row_indices, col_indices)]
            plt.title(f'{title} (Random Sampling {rows}x{cols})')
            plt.imshow(sampled_weights, cmap='viridis')
        else:
            plt.title(title)
            plt.imshow(weights, cmap='viridis')
        
        plt.colorbar()
        plt.savefig(os.path.join(save_dir, f'layer{layer}_weights_heatmap.png'))
        plt.close() 
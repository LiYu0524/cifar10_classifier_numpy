import numpy as np
import pickle

class ActivationFunction:
    """激活函数基类"""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x, dout):
        raise NotImplementedError

class Sigmoid(ActivationFunction):
    """Sigmoid激活函数"""
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def backward(self, x, dout):
        fx = self.forward(x)
        return dout * fx * (1 - fx)

class ReLU(ActivationFunction):
    """ReLU激活函数"""
    def forward(self, x):
        return np.maximum(0, x)
    
    def backward(self, x, dout):
        dx = dout.copy()
        dx[x <= 0] = 0
        return dx

class Tanh(ActivationFunction):
    """Tanh激活函数"""
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, x, dout):
        fx = self.forward(x)
        return dout * (1 - fx * fx)

class SoftmaxWithCrossEntropy:
    """Softmax + 交叉熵损失"""
    def forward(self, x, t):
        """
        x: 模型输出
        t: 目标值（one-hot编码）
        """
        self.t = t
        self.x_shape = x.shape
        
        # 为数值稳定性减去最大值
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.y = exp_x / np.sum(exp_x, axis=1, keepdims=True)  # softmax
        
        if self.t.ndim == 1:  # 如果标签是索引而不是one-hot
            self.t_one_hot = np.zeros_like(self.y)
            self.t_one_hot[np.arange(self.t.shape[0]), self.t] = 1
        else:
            self.t_one_hot = self.t
        
        # 计算交叉熵损失
        loss = -np.sum(self.t_one_hot * np.log(self.y + 1e-10)) / self.x_shape[0]
        return loss
    
    def backward(self):
        """返回损失关于输入的梯度"""
        dx = (self.y - self.t_one_hot) / self.x_shape[0]
        return dx

class ThreeLayerNet:
    """三层神经网络"""
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, 
                 activation='relu', weight_init_std=0.01):
        """
        初始化网络参数
        
        参数:
        - input_size: 输入大小
        - hidden_size1: 第一个隐藏层大小
        - hidden_size2: 第二个隐藏层大小
        - output_size: 输出大小
        - activation: 激活函数类型 ('relu', 'sigmoid', 'tanh')
        - weight_init_std: 权重初始化标准差
        """
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

        # 设置激活函数
        if activation.lower() == 'relu':
            self.activation = ReLU()
        elif activation.lower() == 'sigmoid':
            self.activation = Sigmoid()
        elif activation.lower() == 'tanh':
            self.activation = Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        self.loss_function = SoftmaxWithCrossEntropy()
        
        # 中间数据
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        self.z3 = None

    def predict(self, x):
        """前向传播"""
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        z1 = np.dot(x, W1) + b1
        a1 = self.activation.forward(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = self.activation.forward(z2)
        z3 = np.dot(a2, W3) + b3
        
        return z3
    
    def loss(self, x, t, reg_lambda=0.0):
        """计算损失（包括L2正则化）"""
        z3 = self.predict(x)
        
        # 计算交叉熵损失
        loss = self.loss_function.forward(z3, t)
        
        # 添加L2正则化
        if reg_lambda > 0:
            W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
            reg_loss = 0.5 * reg_lambda * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
            loss += reg_loss
            
        return loss
        
    def accuracy(self, x, t):
        """计算准确率"""
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
            
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def forward(self, x):
        """前向传播并存储中间值"""
        self.x = x
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        
        self.z1 = np.dot(x, W1) + b1
        self.a1 = self.activation.forward(self.z1)
        self.z2 = np.dot(self.a1, W2) + b2
        self.a2 = self.activation.forward(self.z2)
        self.z3 = np.dot(self.a2, W3) + b3
        
        return self.z3
        
    def backward(self, t, reg_lambda=0.0):
        """反向传播计算梯度"""
        batch_size = self.x.shape[0]
        
        # 计算softmax与交叉熵的梯度
        dout = self.loss_function.backward()
        
        # 输出层到第二个隐藏层
        dW3 = np.dot(self.a2.T, dout)
        db3 = np.sum(dout, axis=0)
        da2 = np.dot(dout, self.params['W3'].T)
        
        # 第二个隐藏层激活函数
        dz2 = self.activation.backward(self.z2, da2)
        
        # 第二个隐藏层到第一个隐藏层
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        da1 = np.dot(dz2, self.params['W2'].T)
        
        # 第一个隐藏层激活函数
        dz1 = self.activation.backward(self.z1, da1)
        
        # 第一个隐藏层到输入层
        dW1 = np.dot(self.x.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        # 添加L2正则化
        if reg_lambda > 0:
            dW1 += reg_lambda * self.params['W1']
            dW2 += reg_lambda * self.params['W2']
            dW3 += reg_lambda * self.params['W3']
        
        # 存储梯度
        grads = {}
        grads['W1'] = dW1
        grads['b1'] = db1
        grads['W2'] = dW2
        grads['b2'] = db2
        grads['W3'] = dW3
        grads['b3'] = db3
        
        return grads
    
    def save_params(self, file_path):
        """保存模型参数"""
        with open(file_path, 'wb') as f:
            pickle.dump(self.params, f)
    
    def load_params(self, file_path):
        """加载模型参数"""
        with open(file_path, 'rb') as f:
            self.params = pickle.load(f) 
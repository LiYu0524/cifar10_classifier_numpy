import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from model import ThreeLayerNet, TwoLayerNet
from data import load_cifar10

def test_model(model_path, dataset_dir='./cifar10/dataset', 
               input_size=3072, hidden_size=100, hidden_size2=None, 
               output_size=10, activation='relu', model_type='three'):
    """
    测试模型在测试集上的性能
    
    参数:
    - model_path: 模型参数文件路径
    - dataset_dir: 数据集目录
    - input_size: 输入大小
    - hidden_size: 隐藏层大小 (如果是两层网络) 或第一隐藏层大小 (如果是三层网络)
    - hidden_size2: 第二隐藏层大小 (仅用于三层网络)
    - output_size: 输出大小
    - activation: 激活函数类型
    - model_type: 模型类型 ('two' 或 'three')
    
    返回:
    - 测试准确率
    """
    # 加载测试数据
    print("加载CIFAR-10数据集...")
    _, (X_test, y_test) = load_cifar10(dataset_dir, normalize=True, flatten=True, one_hot=False)
    
    # 创建模型
    if model_type == 'two':
        print(f"创建两层神经网络 (hidden_size={hidden_size}, activation={activation})...")
        model = TwoLayerNet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            activation=activation
        )
    else:  # 默认为三层网络
        if hidden_size2 is None:
            hidden_size2 = hidden_size  # 如果未指定，默认与第一层相同
        print(f"创建三层神经网络 (hidden_size1={hidden_size}, hidden_size2={hidden_size2}, activation={activation})...")
        model = ThreeLayerNet(
            input_size=input_size,
            hidden_size1=hidden_size,
            hidden_size2=hidden_size2,
            output_size=output_size,
            activation=activation
        )
    
    # 加载模型参数
    print(f"从 {model_path} 加载模型参数...")
    model.load_params(model_path)
    
    # 在测试集上评估
    print("在测试集上评估模型性能...")
    test_acc = model.accuracy(X_test, y_test)
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 可视化随机样本的预测
    visualize_predictions(model, X_test, y_test)
    
    # 计算并可视化混淆矩阵
    confusion_matrix(model, X_test, y_test)
    
    return test_acc

def visualize_predictions(model, X, y, n_samples=10, save_path='./cifar10/results/test_predictions.png'):
    """
    可视化模型在随机样本上的预测结果
    
    参数:
    - model: 神经网络模型
    - X: 输入数据 (已展平)
    - y: 真实标签
    - n_samples: 样本数量
    - save_path: 图像保存路径
    """
    # CIFAR-10类别名称
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    # 随机选择样本
    indices = np.random.choice(X.shape[0], n_samples, replace=False)
    X_samples = X[indices]
    y_samples = y[indices]
    
    # 获取预测结果
    predictions = model.predict(X_samples)
    pred_classes = np.argmax(predictions, axis=1)
    
    # 绘制图像
    plt.figure(figsize=(12, 4))
    for i in range(n_samples):
        plt.subplot(2, 5, i+1)
        # 将展平的图像重新塑形为32x32x3
        img = X_samples[i].reshape(32, 32, 3)
        plt.imshow(img)
        plt.title(f"True: {class_names[y_samples[i]]}\nPred: {class_names[pred_classes[i]]}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # 确保保存路径目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"预测可视化已保存至 {save_path}")

def confusion_matrix(model, X, y, save_path='./cifar10/results/confusion_matrix.png'):
    """
    计算并可视化混淆矩阵
    
    参数:
    - model: 神经网络模型
    - X: 输入数据
    - y: 真实标签
    - save_path: 图像保存路径
    """
    # CIFAR-10类别名称
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    # 获取预测结果
    predictions = model.predict(X)
    pred_classes = np.argmax(predictions, axis=1)
    
    # 计算混淆矩阵
    conf_matrix = np.zeros((10, 10), dtype=int)
    for true_label, pred_label in zip(y, pred_classes):
        conf_matrix[true_label, pred_label] += 1
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 设置坐标轴
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 标注数值
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵已保存至 {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试神经网络在CIFAR-10数据集上的性能')
    
    parser.add_argument('--model_path', type=str, default='./cifar10/results/best_model.pkl',
                        help='模型参数文件路径')
    parser.add_argument('--dataset_dir', type=str, default='./cifar10/dataset',
                        help='数据集目录')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='隐藏层大小')
    parser.add_argument('--model_type', type=str, default='two',
                        help='模型类型 (two 或 three)')
    parser.add_argument('--activation', type=str, default='relu',
                        help='激活函数类型 (relu, sigmoid, tanh)')
    
    args = parser.parse_args()
    
    # 加载测试数据
    _, (X_test, y_test) = load_cifar10(args.dataset_dir, normalize=True, flatten=True, one_hot=False)
    
    # 测试模型
    test_acc = test_model(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        input_size=X_test.shape[1],
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation,
        model_type=args.model_type
    )
    
    print(f"测试集准确率: {test_acc:.4f}") 
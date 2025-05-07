from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.X=X
        return np.dot(self.X,self.W)+self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.dot(self.X.T, grad)
        
        # 如果启用权重衰减，添加正则化项的梯度
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        # ================= 计算偏置梯度 =================
        # 公式: dL/db = Σ(grad) along batch维度
        # grad 形状: [batch_size, out_dim]
        # 求和后形状: [1, out_dim] (与b一致)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)

        # ================= 计算输入梯度 =================
        # 公式: dL/dX = grad · W^T
        # grad 形状: [batch_size, out_dim]
        # W.T 形状: [out_dim, in_dim]
        # 矩阵乘法结果形状: [batch_size, in_dim] (与输入X一致)
        dX = np.dot(grad, self.W.T)

        return dX  # 传递给前一层
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels      # 输入通道数
        self.out_channels = out_channels    # 输出通道数
        self.kernel_size = kernel_size      # 卷积核大小
        self.stride = stride                # 步长
        self.weight = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))  # 权重 [out, in, k, k]
        self.bias = np.zeros((out_channels,))                         # 偏置 [out]
        self.grads = {'W' : None, 'b' : None}                        # 梯度存储
        self.params = {'W' : self.weight, 'b' : self.bias}           # 参数集合
        self.weight_decay = weight_decay                             # 是否启用权重衰减
        self.weight_decay_lambda = weight_decay_lambda               # 权重衰减系数

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        batch_size, in_channels, H, W = X.shape
        assert in_channels == self.in_channels, "输入通道数不匹配"
        
        # 计算输出尺寸
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        
        # 初始化输出
        output = np.zeros((batch_size, self.out_channels, H_out, W_out))
        
        # 滑动窗口进行卷积计算
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        # 计算当前窗口的位置
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取输入窗口并计算点积
                        window = X[b, :, h_start:h_end, w_start:w_end]
                        output[b, c_out, h, w] = np.sum(window * self.weight[c_out]) + self.bias[c_out]
        return output


    def backward(self, grads):
        X = self.input  # 使用保存的输入数据
        batch_size, out_channels, H_out, W_out = grads.shape
        _, in_channels, H, W = X.shape

        dX = np.zeros_like(X)
        dW = np.zeros_like(self.weight)
        db = np.sum(grads, axis=(0, 2, 3))

        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        # ... 滑动窗口计算代码 ... 
                        # 计算当前窗口的位置
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # 提取输入窗口
                        window = X[b, :, h_start:h_end, w_start:w_end]
                        
                        # 累加权重梯度 dW
                        dW[c_out] += grads[b, c_out, h, w] * window
                        
                        # 累加输入梯度 dX
                        dX[b, :, h_start:h_end, w_start:w_end] += self.weight[c_out] * grads[b, c_out, h, w]
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.weight
            
        self.grads['W'] = dW
        self.grads['b'] = db
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model            # 关联的模型用于反向传播
        self.max_classes = max_classes
        self.has_softmax = True       # 默认启用softmax
        self.softmax_output = None    # 保存前向传播结果
        self.predicts = None          # 保存原始预测值
        self.labels = None            # 保存标签
        self.batch_size = 0           # 记录批大小

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def _cross_entropy_loss(self, probs, labels):
        """计算交叉熵损失
        Args:
            probs  : [batch_size, D] softmax输出的概率分布
            labels : [batch_size] 真实标签
        Returns:
            loss : scalar
        """
        batch_size = probs.shape[0]
        
        # 获取每个样本对应真实类别的概率（一维数组）
        correct_probs = probs[np.arange(batch_size), labels]  # 形状 (B,)
        
        # 加小量避免log(0)
        epsilon = 1e-12
        correct_log_probs = -np.log(np.maximum(correct_probs, epsilon))  # 对正确类别概率取负对数
        
        # 求平均损失
        loss = np.mean(correct_log_probs)
        
        return loss

    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        assert len(predicts.shape) == 2, "Predicts must be 2D tensor"
        assert len(labels.shape) == 1, "Labels must be 1D tensor"
        assert predicts.shape[0] == labels.shape[0], "Batch size mismatch"

        self.batch_size = predicts.shape[0]
        self.predicts = predicts
        self.labels = labels

        self.softmax_output = softmax(predicts)

        # 计算交叉熵损失
        loss = self._cross_entropy_loss(self.softmax_output, labels)
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        one_hot = np.zeros_like(self.softmax_output)
        one_hot[np.arange(self.batch_size), self.labels] = 1
        self.grads = (self.softmax_output - one_hot) / self.batch_size
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
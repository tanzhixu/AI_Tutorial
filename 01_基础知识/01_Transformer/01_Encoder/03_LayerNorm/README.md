### 介绍
Layer normalization（层归一化）是一种神经网络中常用的归一化技术，用于处理每一层的输入数据。它的主要目的是使得神经网络的每一层输入的分布保持稳定，从而加速训练并提高模型的泛化能力。
### 重要概念：

- 输入归一化： 在神经网络的每一层，Layer normalization对该层的输入进行归一化处理。这意味着对每个样本，在同一层的所有神经元的输出都会被归一化。
- 减少内部协变量偏移： Layer normalization有助于减少内部协变量偏移（Internal Covariate Shift），这是指网络在训练过程中由于参数更新而导致每一层输入分布的变化。通过归一化每一层的输入，Layer normalization使得每一层的激活函数的输入分布更加稳定，有助于加速训练。
- 独立样本归一化： 与Batch normalization不同，Layer normalization是在每个样本的维度上进行归一化，而不是在mini-batch的维度上。这意味着每个样本的输入都是独立归一化的，使得Layer normalization更适用于处理序列数据等每个样本尺寸不同的情况。
- 可学习参数： 与Batch normalization类似，Layer normalization引入了可学习的缩放参数 $\gamma$ 和偏移参数 $\beta$。这些参数允许模型在训练过程中学习适当的输入范围和偏移，从而增加了模型的灵活性。
- 应用范围： Layer normalization通常用于深度神经网络中的各种任务，包括自然语言处理、图像处理和其他序列数据处理任务。它已经被证明在许多情况下能够提高模型的收敛速度和泛化能力。

总的来说，Layer normalization通过对每一层的输入进行归一化处理，有助于减少内部协变量偏移，加速训练，提高模型的泛化能力，特别适用于处理序列数据或每个样本尺寸不同的情况

### 公式
$$
\begin{aligned}
\mathrm{LayerNorm}(x) &=\frac{x-\mu(\mathbf{x})}{\sigma(\mathbf{x})}+\beta \\
\mu(\mathbf{x}) &=\frac{1}{H W} \sum_{i=1}^{H W} x_{i} \\
\sigma(\mathbf{x}) &=\sqrt{\frac{1}{H W} \sum_{i=1}^{H W} (x_{i}-\mu(\mathbf{x}))^{2}}
\end{aligned}
$$

### 均值
$$
\begin{aligned}
\mu(\mathbf{x}) &=\frac{1}{H W} \sum_{i=1}^{H W} x_{i} 
\end{aligned}
$$

### 方差
$$
\begin{aligned}
\\
\sigma(\mathbf{x}) &=\sqrt{\frac{1}{H W} \sum_{i=1}^{H W} (x_{i}-\mu(\mathbf{x}))^{2}}
\end{aligned}
$$

### 归一化
$$
\begin{aligned}
x_{i} &=\frac{x_{i}-\mu(\mathbf{x})}{\sigma(\mathbf{x})}+\beta \\
\end{aligned}
$$


### 代码
```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

# 使用示例
input_tensor = torch.randn(20, 30, 40)
layer_norm = LayerNorm(normalized_shape=40)
output_tensor = layer_norm(input_tensor)
print(output_tensor.shape)
```
            
### 参考
- [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm)


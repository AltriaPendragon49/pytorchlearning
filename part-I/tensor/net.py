import torch
from torch import nn

from functions import add

class NeuralNetwork_1(nn.Module):
    def __init__(self, function):
        super(NeuralNetwork_1, self).__init__()
        self.net = function

    def forward(self, input):
        print(input)
        output = self.net(input[0].item(), input[1].item())    # 取出tensor中的数值再运算，麻烦！
        return output

class NeuralNetwork_2(nn.Module):
    def __init__(self):
        super(NeuralNetwork_2, self).__init__()
        self.weight = torch.tensor([1, 1])

    def forward(self, input):
        output = input @ self.weight.T # 以tensor乘法代替加法
        return output

if __name__ == "__main__":      # 主程序入口
    x = torch.tensor([1, 3])    # 向量
    print(x.shape)

    # 以tensor作输入, 输出
    model_1 = NeuralNetwork_1(add)
    y = model_1(x)              # == model_1.forward([x, y])      
    print(y)

    # 让网络使用tensor的运算, 而不是python3的运算
    model_1 = NeuralNetwork_2()
    y = model_1(x)              
    print(y)
import torch
from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Linear(in_features=2, out_features=1, bias=True)

    def forward(self, input):
        output = self.net(input)
        return output

if __name__ == "__main__":
    model = NeuralNetwork()

    x = torch.tensor([100.3, 6.8])
    y = torch.tensor([193.8])
    learning_rate = 0.00005

    loss_fn = nn.MSELoss()         # 用PyTorch计算出均方损失误差
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)     # 用PyTorch实现随机梯度下降

    for i in range(1000):
        loss = loss_fn(model(x), y)

        optimizer.zero_grad()    # 梯度清零
        loss.backward()
        optimizer.step()        # 参数更新(调整参数)

    print(model.net.weight, model.net.bias)
    print(model(x))
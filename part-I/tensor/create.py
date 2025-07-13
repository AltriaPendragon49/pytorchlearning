import torch

data = [[1, 2],[3, 4]]            # 将list转化为tensor(张量)
x_data = torch.tensor(data)        # 二维tensor也就是矩阵
print(x_data)

# 根据形状, 创建tensor
rand_tensor = torch.rand(2, 3)    # 从区间[0, 1)的均匀分布中抽取的一组随机数
ones_tensor = torch.ones(1, 3)
zeros_tensor = torch.zeros(3, 5)

print("Random Tensor:")
print(rand_tensor)
print("Ones Tensor:")
print(ones_tensor)
print("Zeros Tensor:")
print(zeros_tensor)

# tensor的一些属性
print(f"Shape of tensor: {rand_tensor.shape}")        # f表示格式化字符串，将{}里的内容以字符串的形式输出
print(f"Datatype of tensor: {rand_tensor.dtype}")
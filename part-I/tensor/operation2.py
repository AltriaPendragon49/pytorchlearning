import torch

# tensor乘法(二维tensor乘法就是矩阵乘法)
tensor = torch.ones(2, 3)
tensor[1][0] = -1
print(f'tensor: \n {tensor}')

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
print(f'y1: \n {y1}')
print(f'y2: \n {y2}')

# tensor对应元素乘法(Hadamard product)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
print(f'z1: \n {z1}')
print(f'z2: \n {z2}')

agg = tensor.sum()            # tensor元素求和
print(agg)                    # 标量
agg_item = agg.item()        # 取出tensor中的数值
print(agg_item, type(agg_item))

tensor.add_(5)                # tensor加法的一种形式，类似于python3 a += 1
print(tensor)
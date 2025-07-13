import torch

tensor = torch.eye(4, 4)                # 根据形状创建对角tensor
print('First row: ', tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[:, -1])
tensor[:,1] = 0 # 将第二列的元素设置为0
print(tensor)

t1 = torch.cat([tensor, tensor], dim=0)    # 沿着某一维度拼接tensor(0表示行，1表示列，-1表示倒数第一维度，-2表示倒数第二维度)
print(t1, t1.shape)
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Toy dataset 每一个输入x对应一个输出y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 此处搭建线性回归模型


# 此处声明Loss和optimizer


# 此处进行模型训练


# 将x、y、以及预测的y画在二维坐标轴上
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# 保存模型
# torch.save(model.state_dict(), 'model.ckpt')
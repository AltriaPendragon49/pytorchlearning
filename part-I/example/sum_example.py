import torch
import time


if __name__ == "__main__":      # 主程序入口
    x = torch.rand(10000, dtype=torch.float64).cuda()    # 向量
    weight = torch.ones(10000, dtype=torch.float64).cuda()
    print(x.shape)

    # 使用两个tensor相乘实现矩阵某一维度求和
    tensor_mul_start = time.time()
    for ite in range(500):
        y = x @ weight.T
        # print("iter:", ite, "result:", y)
    tensor_mul_end = time.time()              
    print(tensor_mul_end - tensor_mul_start)
    print("tensor_mul", y)

    # 使用pytorch集成的求和功能
    pytorch_sum_start = time.time()
    for ite in range(500):
        y = torch.sum(x)
        # print("iter:", ite, "result:", y)
    pytorch_sum_end = time.time()              
    print(pytorch_sum_end - pytorch_sum_start)
    print("pytorch_sum", y)
    

    # 普通迭代式加法
    normal_add_start = time.time()
    for ite in range(500):
        y = torch.tensor(0, dtype=torch.float64).cuda() 
        for i in range(list(x.shape)[0]):
            y += x[i]
        # print("iter:", ite, "result:", y)
    normal_add_end = time.time()
    print(normal_add_end - normal_add_start)
    print("normal_add", y)
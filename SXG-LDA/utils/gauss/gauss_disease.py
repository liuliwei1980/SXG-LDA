import numpy as np
import pandas as pd

# 高斯谱核相似性

# nm: 表示miRNA的数量，值为240。
nm = 240
# nd: 表示疾病的数量，值为154。
nd = 154
# A: 一个240x154的矩阵，用于存储miRNA和疾病之间的连接信息。
A = np.loadtxt('l-d.txt')

"""
Getgauss_disease(adjacentmatrix, nd): 这个函数计算疾病之间的高斯谱核相似性，adjacentmatrix是一个形状为(nd, nd)的数组，
表示疾病之间的连接信息，nd是疾病的数量。函数通过计算欧氏距离来度量疾病之间的相似性，并使用高斯核函数来将距离转换为相似性得分。
最后，返回一个(nd, nd)的数组，表示疾病之间的高斯谱核相似性矩阵。
"""
def Getgauss_disease(adjacentmatrix, nd):
    KD = np.zeros((nd, nd), dtype=np.float32)
    gamma = 1
    sum_norm = 0
    for i in range(nd):
        sum_norm = np.linalg.norm(adjacentmatrix[:, i]) ** 2 + sum_norm
    gamma = gamma / (sum_norm / nd)

    for i in range(nd):
        for j in range(nd):
            if j <= i:
                KD[i, j] = np.exp(-gamma * (np.linalg.norm(adjacentmatrix[:, i] - adjacentmatrix[:, j])) ** 2)
    KD = KD + KD.T - np.eye(nd)
    return KD


# 通过调用上述函数，得到疾病之间的高斯谱核相似性矩阵KD：
# KD: 是一个154x154的数组，表示疾病之间的高斯谱核相似性。
KD = Getgauss_disease(A, nd)

# 保存 KD 矩阵到文件
np.savetxt('KD_matrix.txt', KD)

# # 保存 KD 矩阵到 Excel 文件
# df = pd.DataFrame(KD)
# df.to_excel('KD_matrix.xlsx', index=False)
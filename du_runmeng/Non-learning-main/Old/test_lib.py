import numpy as np
from scipy.linalg import orth
import torch

N = 3

##生成一个固定大小的正交矩阵
def genorthogonal_matrix (n):
    # 生成一个随机矩阵
    random_matrix = np.random.rand(n,n)
    # 使用 orth 函数生成正交矩阵
    orthogonal_matrix = orth(random_matrix)
    return orthogonal_matrix

##混淆矩阵的聚合
def mix_aggregation(orthogonal_M_,confuse_gradents):
    confuse_gradents_T=np.transpose(confuse_gradents) ##转置
    return np.dot(confuse_gradents_T,orthogonal_M_)

#混淆矩阵的聚合测试
def test_aggregation():
    M1=genorthogonal_matrix(N)  ###生成一个可逆矩阵
    print("M1 dot M1.T:", np.dot(M1, M1.T))                         
    one_vector= np.ones(N)
    orthogonal_M_=np.dot(M1,one_vector)
    G= np.random.randint(0, 10, size=(N, N))### N*N 参数矩阵 整数
    confuse_gradents=np.dot(M1,G)
    print("生成的随机梯度矩阵为:",G)
    print("生成的混淆梯度矩阵为:",confuse_gradents)
    print("M·1向量为",orthogonal_M_)
    print("混淆的聚合值:",mix_aggregation(orthogonal_M_,confuse_gradents))
    
    
test_aggregation()
x = 1
# a = np.array()
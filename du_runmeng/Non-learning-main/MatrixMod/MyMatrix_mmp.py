
import numpy as np
from copy import deepcopy
import pickle
import sympy
import os
import galois
import time
from utils_mmp import A_Matrix_path, precisions, preci

precision, matrix_P = precisions[preci]
MMP = galois.GF(matrix_P)

def generate_matrics(n=784, p=matrix_P):
    A_Matrix = []
    A_Matrix_ = []
    def generate_invertible_matrix_mod_p(n, p=matrix_P):
        A = np.random.randint(1, p, size=(n, n))
        return A
    numpy_matrix = generate_invertible_matrix_mod_p(n)
    A = MMP(numpy_matrix)
    A_inv = np.linalg.inv(A)
    
    to_saveA = A.tolist()
    to_saveA_inv = A_inv.tolist()
    A_Matrix.append(to_saveA)
    A_Matrix_.append(to_saveA_inv)
    with open(A_Matrix_path, "wb") as f:
        pickle.dump((A_Matrix, A_Matrix_), f)

def load_matrix():
    with open(A_Matrix_path, "rb") as f:
        return pickle.load(f)
        # ListA, ListA_ = pickle.load(f)
        # assert len(ListA) == len(ListA_)
        # A_Matrix = [MMP(ListA[i]) for i in range(len(ListA))]
        # A_Matrix_ = [MMP(ListA_[i]) for i in range(len(ListA))]
        # return A_Matrix, A_Matrix_
        

def expand_matrix(nums):
    A_Matrix, A_Matrix_ = load_matrix()
    assert len(A_Matrix)>0 and len(A_Matrix_)>0 and len(A_Matrix)==len(A_Matrix_)
    if(len(A_Matrix) == nums):
        return A_Matrix, A_Matrix_
    tmp_A_Matrix = [A_Matrix[0] for i in range(nums)]
    tmp_A_Matrix_ = [A_Matrix_[0] for i in range(nums)]
    with open(A_Matrix_path, "wb") as f:
        pickle.dump((tmp_A_Matrix, tmp_A_Matrix_), f)
    return tmp_A_Matrix,tmp_A_Matrix_

def test2():
    generate_matrics()
    expand_matrix(60000)
    A_Matrix, A_Matrix_ = load_matrix()

    A = MMP(A_Matrix[0])
    A_ = MMP(A_Matrix_[0])
    
    start_time = time.time()
    s = A @ A_
    endtime = time.time()
    print(f"Executiong time {endtime-start_time} seconds")
    print(s)

def test3():
    A_Matrix, A_Matrix_ = load_matrix()
    a = MMP(A_Matrix[0])
    a_ = MMP(A_Matrix_[0])
    Wh1 = MMP(np.random.randint(1,1000,size=(784, 1)))
    Wh1_ = a @ (a_ @ Wh1)
    return 

if __name__ == "__main__":
    n = 784
    test2()
    test3()
    
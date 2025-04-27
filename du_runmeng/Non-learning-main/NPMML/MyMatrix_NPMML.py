
import numpy as np
from copy import deepcopy
import pickle
import sympy
import os
import time
from utils_NPMML import A_Matrix_float_path


def generate_matrics_float(n=784):
    A_Matrix = []
    A_Matrix_ = []
    
    def generate_invertible_integer_matrix(n, min_val=-100, max_val=100):
        while True:
            matrix = np.array([[np.random.randint(min_val, max_val) for _ in range(n)] for _ in range(n)])
            if np.linalg.matrix_rank(matrix) == n:
                return matrix
    
    numpy_matrix = generate_invertible_integer_matrix(n)
    numpy_inverse = np.linalg.inv(numpy_matrix)
    A_Matrix.append(numpy_matrix)
    A_Matrix_.append(numpy_inverse)
    with open(A_Matrix_float_path, "wb") as f:
        pickle.dump((A_Matrix, A_Matrix_), f)

def load_matrix_float():
    with open(A_Matrix_float_path, "rb") as f:
        return pickle.load(f)

def expand_matrix_float(nums):
    A_Matrix, A_Matrix_ = load_matrix_float()
    assert len(A_Matrix)>0 and len(A_Matrix_)>0 and len(A_Matrix)==len(A_Matrix_)
    if(len(A_Matrix) == nums):
        return A_Matrix, A_Matrix_
    tmp_A_Matrix = [A_Matrix[0] for i in range(nums)]
    tmp_A_Matrix_ = [A_Matrix_[0] for i in range(nums)]
    with open(A_Matrix_float_path, "wb") as f:
        pickle.dump((tmp_A_Matrix, tmp_A_Matrix_), f)
    return tmp_A_Matrix,tmp_A_Matrix_

def test2():
    generate_matrics_float()
    expand_matrix_float(60000)
    A_Matrix, A_Matrix_ = load_matrix_float()

    A = A_Matrix[0]
    A_ = A_Matrix_[0]
    
    start_time = time.time()
    s = A @ A_
    endtime = time.time()
    print(f"Executiong time {endtime-start_time} seconds")
    print(s)

def test3():
    A_Matrix, A_Matrix_ = load_matrix_float()
    a = A_Matrix[0]
    a_ = A_Matrix_[0]
    Wh1 = np.random.randint(1,1000,size=(784, 1))
    Wh1_ = a @ (a_ @ Wh1)
    return 

if __name__ == "__main__":
    n = 784
    test2()
    test3()
    
import galois
import numpy as np
from MyMatrix import load_matrix,MyMatrix
import time


A_Matrix, A_Matrix_ = load_matrix()
A = A_Matrix[0].matrix
A_ = A_Matrix_[0].matrix

matrix_P = 262139

# Define the Galois field GF(p)
GF = galois.GF(matrix_P)

# Define your matrices using the Galois field
A = GF(A)
B = GF(A_)


# Perform the matrix multiplication

start_time = time.time()
C = A @ B
end_time = time.time()

execution_time = end_time - start_time
print(f"The function took {execution_time} seconds to execute.")

print(C)
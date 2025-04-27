import numpy as np
import os
import sympy
import pickle

# from multiprocessing import Pool

# from MyMatrix import MyMatrix,matrix_P
# Best performence in Ubuntu.
num_of_client = 2  # Number of clients
# compute_pool = Pool(num_of_client)
# R = 10 * (60000 // num_of_client)  # Number of rounds to run
# batch_size = 6 * num_of_client
batch_size = 6 * num_of_client
epochs = 3
R = epochs * (batch_size // num_of_client)  # Number of rounds to run

cur_dir = os.path.dirname(__file__)
dataset_path = os.path.join(cur_dir, './dataset/')
enc_data_path = os.path.join(cur_dir, "./parameters/enc_data")
# enc_data_path = "./parameters/enc_data_test"
A_Matrix_path = os.path.join(cur_dir, "./parameters/A_Matrix")  
A_Matrix_float_path = os.path.join(cur_dir, "./parameters/A_Matrix_float")

  
Precision = 10000

# The ES and the CSP can hold the Gama simultaneously.
Gama = 2
Gama_ = 0.5


from Pailler import *
CSP_PA = PaillierCreator()

def test():
    cipher = CSP_PA.encrypt(8)
    print("cipher:", cipher)
    plain = CSP_PA.decrypt_ciper(cipher)
    print("plain:", plain)

def mod_pow(base, exponent, modulus):
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent // 2
        base = (base * base) % modulus
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# a = 1
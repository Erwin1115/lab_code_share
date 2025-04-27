import numpy as np
import sympy
import pickle
import os
# from MyMatrix import MyMatrix,matrix_P

num_of_client = 50  # Number of clients
batch_size = 6 * num_of_client
epochs = 5
R = epochs * (batch_size // num_of_client)  # Number of rounds to run

malicious_rate = 0.3

cur_dir = os.path.dirname(__file__)
dataset_path = os.path.join(cur_dir, './dataset/')
enc_data_path = os.path.join(cur_dir, "./parameters/enc_data")
# enc_data_path = "./parameters/enc_data_test"
A_Matrix_path = os.path.join(cur_dir, "./parameters/A_Matrix")
A_Matrix_float_path = os.path.join(cur_dir, "./parameters/A_Matrix_float")
pickle_data_path = os.path.join(cur_dir, "./parameters/train_data")
baseline_reuslt_path = os.path.join(cur_dir, "result/base_line_minist")


# from Pailler import *
# CSP_PA = PaillierCreator()

# def test():
#     cipher = CSP_PA.encrypt(8)
#     print("cipher:", cipher)
#     plain = CSP_PA.decrypt_ciper(cipher)
#     print("plain:", plain)

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
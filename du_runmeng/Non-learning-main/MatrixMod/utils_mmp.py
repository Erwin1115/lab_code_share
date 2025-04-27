import numpy as np
import os
import sympy
import pickle
# from MyMatrix import MyMatrix,matrix_P

num_of_client = 10  # Number of clients
R = 10 * (60000 // num_of_client)  # Number of rounds to run

cur_dir = os.path.dirname(__file__)
dataset_path = os.path.join(cur_dir, './dataset/')
enc_data_path = os.path.join(cur_dir, "./parameters/enc_data")
# enc_data_path = "./parameters/enc_data_test"
A_Matrix_path = os.path.join(cur_dir, "./parameters/A_Matrix")
A_Matrix_float_path = os.path.join(cur_dir, "./parameters/A_Matrix_float")


prime_20bit = 262139
prime_32bit = 2147483647
prime_64bit = 9223372036854775783
prime_128bit = 340282366920938463463374607431768211507
prime_256bit = 115792089237316195423570985008687907853269984665640564039457584007913129639747

precisions ={
    14: (10000, prime_20bit)
}
  
preci = 14

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
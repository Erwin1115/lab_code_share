import numpy as np
import sympy
import pickle
# from MyMatrix import MyMatrix,matrix_P

num_of_client = 3  # change to your desired value

dataset_path = './dataset/'
enc_data_path = "./parameters/enc_data"
# enc_data_path = "./parameters/enc_data_test"
A_Matrix_path = "./parameters/A_Matrix"
A_Matrix_float_path = "./parameters/A_Matrix_float"


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


# a = 1
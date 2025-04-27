import numpy as np
import os
import sympy
import pickle

# from multiprocessing import Pool

# from MyMatrix import MyMatrix,matrix_P
# Best performence in Ubuntu.
num_of_client = 10  # Number of clients
# compute_pool = Pool(num_of_client)
batch_size = 60
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
    start = time.time()
    cipher = CSP_PA.encrypt(8)
    end = time.time()
    print("cipher:", cipher)
    print(f"Encrypting one number used {end - start} seconds")
    
    start = time.time()
    plain = CSP_PA.decrypt(cipher)
    end = time.time()
    print(f"Decrypting one number used {end - start} seconds")
    print("plain:", plain)


def test_en_decrypt_matrix():
    size = (128,784)
    a = np.random.randint(1, 100, size)
        
    
    start = time.time()
    cipher = CSP_PA.encryMatrix(a)
    end = time.time()
    # print("cipher:", cipher)
    print(f"Encrypting {size} used {end - start} seconds")
    print(f"Encrypting each used {(end - start)/size} seconds")
    
    start = time.time()
    plain = CSP_PA.decryMatrix(cipher)
    end = time.time()
    print(f"Decrypting {size} used {end - start} seconds")
    # print("plain:", plain)
    
    return


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

if __name__ == "__main__":
    # test()
    test_en_decrypt_matrix()

# a = 1
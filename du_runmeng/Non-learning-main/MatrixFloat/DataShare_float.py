import torch
import numpy as np
from torchvision import datasets, transforms
import random
import os
from PIL import Image
import pickle

from utils_float import mod_pow, CSP_PA
from utils_float import dataset_path, enc_data_path
from utils_float import Precision, Gama, batch_size
from MyMatrix_float import load_matrix_float

from tqdm import tqdm
import time



def expand_labels(lables):
    result = []
    for i in range(len(lables)):
        labels_tmp = np.full((10,1), (0.01)*Precision) 
        labels_tmp[lables[i]][0] = (0.99)*Precision
        result.append(labels_tmp)
    return np.array(result)

def load_data(type="FASHION"):
    # Define a transform to convert the data to tensor
    transform = transforms.ToTensor()

    if type == "MINIST":
        # Download and load the training and test data
        trainset = datasets.MNIST(dataset_path, download=True, train=True, transform=transform)
        testset = datasets.MNIST(dataset_path, download=True, train=False, transform=transform)
    elif type == "FASHION":
        trainset = datasets.FashionMNIST(dataset_path, download=True, train=True, transform=transform)
        testset = datasets.FashionMNIST(dataset_path, download=True, train=False, transform=transform)
     
    # Access the data and the labels
    train_data = trainset.data.numpy().reshape(-1, 28*28)
    train_labels = trainset.targets.numpy()

    test_data = testset.data.numpy().reshape(-1, 28*28)
    test_labels = testset.targets.numpy()
    
    tmean = np.mean(train_data)
    tstd = np.std(test_data)

    train_data = (train_data - tmean) / tstd
    test_data = (test_data - tmean) / tstd
    
    train_labels = expand_labels(train_labels)
    # test_labels = expand_labels(test_labels)
    
    return train_data,train_labels,test_data,test_labels



# Define a custom function.
def pailliar_y(x):
    assert type(x) is int or type(x) is np.int32
    return mod_pow(CSP_PA.g,x,CSP_PA.public_key.nsquare)
# Vectorize the function.
vfunc_pailliar = np.vectorize(pailliar_y)


def encrypt_data(train_data,train_labels,test_data,test_labels):
    A_Matrix, A_Matrix_ = load_matrix_float()

    train_data_list = train_data.tolist()
    train_label_list = train_labels.tolist()
    train_data = np.array(train_data_list)
    
    enc_result = []
    print("Encrypting data:")
    for i in tqdm(range(batch_size)):
        data = train_data[i].T
        label = np.array(train_label_list[i])
        enc_matrix = A_Matrix[i] @ data
        
        # TODO: Hardcode the shape of Y here
        # labels_tmp = np.full((10,1), 0.01) 
        # labels_tmp[label][0] = 0.99
        # enc_y = (labels_tmp * Precision).astype(int)
        # enc_y = vfunc_pailliar(enc_y)
        enc_y = CSP_PA.encryMatrix(label)
        
        enc_result.append((i, enc_matrix, enc_y))
        a = 1
    with open(enc_data_path, "wb") as f:
        pickle.dump(enc_result, f)
    print("All encrypted")
    return enc_result


def encrypt_data_test(train_data,train_labels,test_data,test_labels):
    A_Matrix, A_Matrix_ = load_matrix_float()

    train_data_list = train_data.tolist()
    train_label_list = train_labels.tolist()
    train_data = np.array(train_data_list)
    
    enc_result = []
    start_time = time.time()
    print("Encrypting data:")
    for i in tqdm(range(batch_size)):
        data = train_data[i].T
        label = np.array(train_label_list[i])
        enc_matrix = A_Matrix[i] @ data
        
        # TODO: Hardcode the shape of Y here
        # labels_tmp = np.full((10,1), 0.01) 
        # labels_tmp[label][0] = 0.99
        # enc_y = (labels_tmp * Precision).astype(int)
        # enc_y = vfunc_pailliar(enc_y)
        # enc_y = CSP_PA.encryMatrix(label)
        
        # enc_result.append((i, enc_matrix))
        a = 1
    end_time = time.time()
    print(f"Encrypting {batch_size} Matrixs used {end_time-start_time} seconds. {(end_time-start_time)/batch_size} per matrix")
    print("All encrypted")
    return enc_result




def verify_dec_data(enc_result, train_data):
    A_Matrix, A_Matrix_ = load_matrix_float()
    print("Verifying encrypted data:")
    for i, enc_data, enc_label in tqdm(enc_result):
        assert (A_Matrix_[i] @ enc_data).any() == train_data[i].any()
    print("All Verification passed")
    return None



def load_enc_data():
    if(os.path.exists(enc_data_path)):
        with open(enc_data_path, "rb") as f:
            return pickle.load(f)
    else:
        train_data,train_labels,test_data,test_labels = load_data()
        start_time = time.time()
        enc_data =  encrypt_data(train_data,train_labels, None, None)
        end_time = time.time()
        print(f"Encryption time:{end_time-start_time}s")
        a = pickle.dumps(enc_data)
        print(f"Size:{len(a)} Bytes, {len(a)//1024}KB, {len(a)//(1024*1024)}MB, {len(a)//(1024*1024)/1024}GB")
        
        verify_dec_data(enc_data, train_data)
        return enc_data

def test_enc_dec():
    print("Loading encrypted data:")
    enc_data = load_enc_data()

if __name__ == "__main__":
    test_enc_dec()

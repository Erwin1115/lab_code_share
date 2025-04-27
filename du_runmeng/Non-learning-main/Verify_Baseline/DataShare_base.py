import torch
import numpy as np
from torchvision import datasets, transforms
import random
import os
from PIL import Image
import pickle

from utils_base import mod_pow
from utils_base import dataset_path, pickle_data_path
from utils_base import batch_size
import time
from tqdm import tqdm

def load_data(type = "MINIST"):
    if(os.path.exists(pickle_data_path)):
        with open(pickle_data_path, "rb") as f:
            return pickle.load(f)
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

    # Convert numpy arrays to MyMatrix
    # train_data = MyMatrix(train_data.tolist())
    # test_data = MyMatrix(test_data.tolist())
    
    # result = [(i, train_data[i], train_labels[i]) for i in range(len(train_data))]
    # return result,test_data,test_labels
    
    train_data = train_data[:batch_size]
    train_labels = train_labels[:batch_size]
    test_data = test_data[:5000]
    test_labels = test_labels[:5000]
    
    with open(pickle_data_path, "wb") as f:
        pickle.dump((train_data,train_labels,test_data,test_labels), f)

    return train_data,train_labels,test_data,test_labels




if __name__ == "__main__":
    train_data,train_labels,test_data,test_labels = load_data()
    # train_data,train_labels,test_data,test_labels = load_data(type="FASHION")
    a = 1 
import torch
import numpy as np
from torchvision import datasets, transforms
import random
import os
from PIL import Image
import pickle

from MyMatrix import MyMatrix, load_matrix, load_matrix_float
from utils import mod_pow, CSP_PA
from utils import dataset_path, enc_data_path

from tqdm import tqdm

def load_data():
    # Define a transform to convert the data to tensor
    transform = transforms.ToTensor()

    # Download and load the training and test data
    trainset = datasets.MNIST(dataset_path, download=True, train=True, transform=transform)
    testset = datasets.MNIST(dataset_path, download=True, train=False, transform=transform)

    # Access the data and the labels
    train_data = trainset.data.numpy().reshape(-1, 28*28)
    train_labels = trainset.targets.numpy()

    test_data = testset.data.numpy().reshape(-1, 28*28)
    test_labels = testset.targets.numpy()

    # Convert numpy arrays to MyMatrix
    # train_data = MyMatrix(train_data.tolist())
    # test_data = MyMatrix(test_data.tolist())

    return train_data,train_labels,test_data,test_labels

def split_dataset(X, N):
    # Split the dataset into N parts
    return [MyMatrix(part.tolist()) for part in np.array_split(X.matrix, N)]


def initialize_random_integers(N, low=1, high=1e18):
    # Initialize N random integers
    # R = np.random.uniform(low, high, N)
    return [random.randint(low, high) for i in range(N)]


def calculate_products(X_parts, R, r_es):
    # Calculate r_es * R[j] * X[j] for each part of the dataset
    return [(r_es * R[j]) * part for j, part in enumerate(X_parts)]

# Load your dataset
# train_data = ...



# r_es = np.random.uniform(1, N)  # change to your desired value
# r_es = random.getrandbits(1024)  # change to your desired value

# X_parts = split_dataset(train_data, num_of_client)
# Y_parts = split_dataset(train_labels, num_of_client)
# R = initialize_random_integers(num_of_client,1,N)
# results = calculate_products(X_parts, R, r_es)

def encrypt_data(train_data,train_labels,test_data,test_labels):
    A_Matrix, A_Matrix_ = load_matrix_float()
    
    enc_result = []
    
    train_data_list = train_data.tolist()
    train_label_list = train_labels.tolist()
    train_data = np.array(train_data_list)

    
    # exit()
    print("Encrypting data:")
    for i in tqdm(range(len(train_data_list)//1000)):
        data = train_data[i].T
        label = train_label_list[i]
        np.dot(A_Matrix_[i], np.dot(A_Matrix[i], np.array([[i+1 for i in range(784)] for j in range(784)])))
        np.dot(A_Matrix_[i], np.dot(A_Matrix[i], data))
        
        enc_result.append((i, np.dot(A_Matrix[i], data), mod_pow(CSP_PA.g,label,CSP_PA.public_key.nsquare)))

        # from PIL import Image
        # import numpy as np
        # # Assuming 'data' is your 28x28 list of integers
        # data = [data_[i*28:i*28+28] for i in range(28)]

        # # Convert the data to a numpy array and reshape to the image dimensions
        # image_data = np.array(data, dtype=np.uint8).reshape((28, 28))

        # # Create an image from the array
        # image = Image.fromarray(image_data)

        # # Save the image
        # image.save(os.path.join('images', f'{i}.png'))

        a = 1
    with open(enc_data_path, "wb") as f:
        pickle.dump(enc_result, f)
    print("All encrypted")
    return enc_result

def verify_dec_data(enc_result, train_data):
    A_Matrix, A_Matrix_ = load_matrix_float()
    print("Verifying encrypted data:")
    for i, enc_data, enc_label in tqdm(enc_result):
        assert (np.dot(A_Matrix_[i], enc_data)) == train_data[i]
    print("All Verification passed")
    return None 

def load_enc_data(train_data,train_labels):
    if(os.path.exists(enc_data_path)):
        with open(enc_data_path, "rb") as f:
            return pickle.load(f)
    else:
        return encrypt_data(train_data,train_labels, None, None)

def test_enc_dec():
    train_data,train_labels,test_data,test_labels = load_data()
    print("Loading data:")
    enc_data = load_enc_data(train_data,train_labels)
    verify_dec_data(enc_data, train_data)

if __name__ == "__main__":
    test_enc_dec()
import socket
import pickle
import struct
import numpy as np
from sympy import invert
from DataShare_float import load_data, load_enc_data
from utils_float import sigmoid,R
from utils_float import Gama_, Gama, Precision, num_of_client
from utils_float import CSP_PA, mod_pow

import argparse

from multiprocessing import Process, Manager

print("Loadding data ...")
# train_data,train_labels,test_data,test_labels = load_enc_data()
EncData = load_enc_data()
print("Loading finished")


# Define a custom function.
def pailliar_enc_(x):
    assert type(x) is int or type(x) is np.int32
    return CSP_PA.object_to_ciper(CSP_PA.encrypt(int(x)))
# Vectorize the function.
pailliar_enc = np.vectorize(pailliar_enc_)

def pailliar_mul_(x,y):
    return mod_pow(x, y, CSP_PA.public_key.nsquare)
pailliar_mul = np.vectorize(pailliar_mul_)

def pailliar_inv_(x):
    return invert(x, CSP_PA.public_key.nsquare)
    # return pow(x, -1 , CSP_PA.public_key.nsquare)
pailliar_inv = np.vectorize(pailliar_inv_)


# Just for test, ES can not decrypt the cipher.
def pailliar_dec_(x):
    return CSP_PA.decrypt_ciper(int(x))
pailliar_dec = np.vectorize(pailliar_dec_)




def Matrix_mul_cipher(A, B, P):
    
    if A.shape[1] != B.shape[0]:
        raise ValueError("The number of columns of the first matrix must equal the number of rows of the second matrix")
    result = [[0 for _ in range(B.shape[1])] for _ in range(A.shape[0])]
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i][j] *=  mod_pow(A[i][k],  B[k][j], P)
                result[i][j] %= P
    
    return result


class ES:
    def __init__(self, ip, port) -> None:
        self.ip = ip
        self.port = port
        pass
    def receive_parameters(self, sock):
        response_size_data = self.recv_all(sock, 4)
        if response_size_data is None:
            print("Server closed connection")
            return None
        response_size = struct.unpack('!I', response_size_data)[0]
        response_data = self.recv_all(sock, response_size)
        if response_data is None:
            print("Server closed connection")
            return None
        parameters = pickle.loads(response_data)
        return parameters

    def send_parameters(self, sock, matrix):
        matrix_data = pickle.dumps(matrix)
        sock.sendall(struct.pack('!I', len(matrix_data)))
        sock.sendall(matrix_data)
        # print(f"Send size: {len(matrix_data)} bytes")

    

    
    
    def forward_prop(self, parameters):
        global train_data, train_labels
        index, hiden_Wh, hiden_Wo = parameters
        hiden_Wh = np.array(hiden_Wh)
        hiden_Wo = np.array(hiden_Wo)
        
        index_, enc_x, enc_y = EncData[index]
        real_enc_x = (enc_x[:,np.newaxis]).T
        
        # 1
        enc_Wh = Gama_ * hiden_Wh
        
        # 2       
        Wo = Gama_ * hiden_Wo
        
        # 3
        O1 = enc_Wh @ real_enc_x.T
        
        # 4
        h = sigmoid(O1)
        
        # 5
        O = Wo @ h
        # TODO: In NPMML, Encrypt Wo with CSP_PA. Statistic the computation cost.
        
        # 5 + 1
        # O = sigmoid(O)
        
        # 6
        
        # O = (Precision * Gama_ * O).astype(int)
        # Apply the function to each element.
        # safe_O = pailliar_enc(O)
        O = (Precision * Gama * O).astype(int)
        safe_O = CSP_PA.encryMatrix(O)
        
        
        
        # backward propagation
        # 7 
        Vt = (-h) * (1 - h)
        
        # 8
        # enc_gama_yj = pailliar_mul(enc_y, Gama)
        enc_gama_yj =  CSP_PA.mulscalarMatrix(enc_y, Gama)
        # enc_gama_yj = pailliar_enc(enc_y)
        left = safe_O - enc_gama_yj
        right = (Precision * h).astype(int).T
        
        # C1 = Matrix_mul_cipher(left, right, CSP_PA.public_key.nsquare)
        C1 = left @ right
        
        # Remember, C1 expand Precision ** 2
        
        
        #9
        # C2_left = (Precision * Vt).astype(int)
        C2_left = Vt
        C2_right = real_enc_x
        C2 = C2_left @ C2_right
        # Remember, C2 expand Precision
        # Remember, C2 expand Precision
                
        # to_Wh_1 = real_train_data_matrix
        # to_Wh_1 = Vt @ real_train_data_matrix
        
        # TODO Encrypt C2 with CSP. Remember to decrypt in CSP.
        
        # 10
        
        C3_left = (enc_gama_yj - safe_O).T
        C3_right = (Precision * Wo).astype(int)
        C3 = C3_left @ C3_right
        # Remember, C2 expand Precision ** 2
        
        # C3_left = (enc_gama_yj * pailliar_inv(safe_O)) % CSP_PA.public_key.nsquare
        # C3_right = Wo
        # C3 = Matrix_mul_cipher(C3_left.T, C3_right, CSP_PA.public_key.nsquare)
        # to_Wh_2 = ((-output_errors)).T @ Wo
        # to_Wh_2 = ((-output_errors) * O * (1-O)).T @ Wo
        # to_Wh_2 = (((train_labels[ind
        # ex]-O)).T @ Wo) * Vt.T
        
        
        # print(f"Index:{index}\nenc_Wh:{Wh}\nenc_Wo{Wo}")
        return index, C1, C2, C3

    def recv_all(self, sock, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.ip, self.port))
            for _ in range(R):
                
                parameters = self.receive_parameters(sock)
                if parameters is None:
                    print("Server closed connection")
                    return
                # print("Received parameters from server: \n{}".format(A))
                # print("Received parameters from server: \n")
                
                result = self.forward_prop(parameters)
                # B = self.generate_random_matrix()
                # C = self.add_matrices(A, B)
                self.send_parameters(sock, result)

# if __name__ == "__main__":
#     # HOST, PORT = "localhost", 9999
#     # Create the parser
#     parser = argparse.ArgumentParser(description="Process an IP address and port.")

#     # Add the arguments
#     parser.add_argument('ip_address', type=str, help='The IP address to process.')
#     parser.add_argument('port', type=int, help='The port to process.')

#     # Parse the arguments
#     args = parser.parse_args()
    
#     HOST = args.ip_address
#     PORT = args.port
#     es = ES(HOST, PORT)
#     es.run()

# if __name__ == "__main__":
def worker():
    HOST, PORT = "localhost", 9999
    es = ES(HOST, PORT)
    es.run()

if __name__ == '__main__':
    processes = []
    with Manager() as manager:
        for i in range(num_of_client):
            # Start a new process to modify the managed dictionary
            p1 = Process(target=worker, args=())
            p1.start()
            processes.append(p1)
        for p in processes:
            p.join()
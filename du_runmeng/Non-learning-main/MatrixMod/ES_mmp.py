import socket
import pickle
import struct
import numpy as np
from DataShare_mmp import load_data
from utils_mmp import sigmoid,R

print("Loadding data ...")
train_data,train_labels,test_data,test_labels = load_data()
print("Loading finished")


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

    
    def forward_prop(self, parameters):
        global train_data, train_labels
        index, Wh, Wo = parameters
        
        real_train_data_matrix = (train_data[index][:,np.newaxis]).T
        
        
        # 1
        Wh = Wh
        
        # 2
        Wo = Wo
        
        # 3
        O1 = Wh @ real_train_data_matrix.T
        
        # 4
        h = sigmoid(O1)
        
        # 5
        O = Wo @ h
        
        # 5 + 1
        # O = sigmoid(O)
        
        # 6 
        O = O
        
        
        
        # backward propagation
        # 7 
        Vt = (-h) * (1 - h)
        
        # 8
        labels_tmp = np.full(O.shape, 0.01) 
        labels_tmp[train_labels[index]][0] = 0.99
        output_errors = O - labels_tmp
        
        to_Wo = output_errors
        # to_Wo = output_errors * O * (1-O)
        # to_Wo = (O - labels_tmp)
        # TODO: here!!!
        to_Wo = to_Wo @ h.T
        
        #9
        # to_Wh_1 = real_train_data_matrix
        to_Wh_1 = Vt @ real_train_data_matrix
        
        # 10
        to_Wh_2 = ((-output_errors)).T @ Wo
        # to_Wh_2 = ((-output_errors) * O * (1-O)).T @ Wo
        # to_Wh_2 = (((train_labels[ind
        # ex]-O)).T @ Wo) * Vt.T
        
        # print(f"Index:{index}\nenc_Wh:{Wh}\nenc_Wo{Wo}")
        return index, to_Wo, to_Wh_1, to_Wh_2
    
    def generate_random_matrix(self, ):
        return np.random.rand(3,3)

    def add_matrices(self, matrix1, matrix2):
        return matrix1 + matrix2

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

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    es = ES(HOST, PORT)
    es.run()
import socket
import pickle
import struct
import numpy as np
from DataShare_base import load_data
from utils_base import sigmoid,R,num_of_client
from utils_base import malicious_rate

from diffiehellman import DiffieHellman
from Cryptodome.Hash import SHA256

from verify_util import *
from multiprocessing import Process, Manager


import time

class ES:
    def __init__(self, ip, port, sigmap) -> None:
        self.ip = ip
        self.port = port
        self.gen_DH_pairs()
        
        print("Loadding data ...")
        self.train_data, self.train_labels, self.test_data, self.test_labels = load_data()
        print("Loading finished")
        self.First_round = True
        self.LR = 0.01
        self.Gama = 2
        self.SIGMAP = sigmap
        pass
    
    
    
    def gen_DH_pairs(self):
        self.c_pk, self.__c_sk = KA.gen()
        self.s_pk, self.__s_sk = KA.gen()
        self.i_pk, self.__i_sk = KA.gen()
    

    
    def gen_gradients_signature(self, index, gradients, masked_gredients:np.ndarray):
        
        prime = DiffieHellman()._prime
        
        Wh,Wo = gradients
        masked_Wh, masked_Wo = masked_gredients
        fwi = flatten(Wh, Wo)
        wi = compress(fwi, BETA)
        fmasked_wi = flatten(masked_Wh, masked_Wo)
        masked_wi = compress(fmasked_wi, BETA)
        
        # wi, masked_wi = compress(gradients, BETA), compress(masked_gredients, BETA)
        
        
        # Pars: wi, ipki
        h = SHA256.new()
        h.update(bytes(fwi.flatten()) + self.i_pk)
        hi = int.from_bytes(h.digest(), "big")
        # Pars: hi, wi, iski
        zi = hi * wi + int.from_bytes(self.__i_sk, "big")
        R = pow(2, masked_wi - wi, prime)
        Lambda = pow(2, wi, prime)
        
        sig = (self.i_pk, hi, zi, R, Lambda)
        self.SIGMAP[index%num_of_client] = (self.i_pk, hi, zi, R, Lambda)
        return sig
    
    
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
        # global train_data, train_labels, First_round
        index, Wh, Wo = parameters
        
        
        # to_check = random.choice(list(self.sig_map.keys()))
        to_check = 0
        if not self.First_round:
            # verify_start = time.time()
            i_pk, hi, zi, R, Lambda = self.SIGMAP[to_check]
        
            prime = DiffieHellman()._prime
            
            fwi = flatten(Wh, Wo)
            wi = compress(fwi, BETA)
            left = pow(2, wi, prime)
            left = (left * pow(Lambda, hi-1, prime)) % prime
            left = (left * int.from_bytes(i_pk, "big")) % prime
            
            right = pow(2, zi, prime)
            # for i in range(20):
            #     print(f"Before iterating!!!!!!!!!!!!!!!!!! num:{i} MAPLEN:{len(self.SIGMAP)}")
            # for o_u in self.SIGMAP:
            for o_u in range(int(num_of_client * (1-malicious_rate))):
                if o_u != to_check:
                    _, _, _, _, tmp_Lambda = self.SIGMAP[o_u]
                    right = (right * tmp_Lambda) % prime
            # verify_end = time.time()
            
            # print(f"RE veirify CSP cost {verify_end-verify_start} seconds")
            # self.output = self.output / self.prec
            if left == right:
                pass
                # logging.info(f"User{self.id} finish verifying server's aggregation, unmask result:{self.output}")
                # logging.info(f"User verifying time: {verify_end-verify_start}s")
                # self.send(pickle.dumps(self.id), host, finish_port)
            else:
                assert 0 == 1
                pass
                # logging.warning(f"User{self.id} receive forged result:{self.output}")
                # logging.warning(f"User{self.id} Left:{left}----------Right:{right}")
                # logging.warning(f"User{self.id} Left-Right:{left-right}")
                # self.send(pickle.dumps(self.id), host, finish_port)
        else:
            self.First_round = False
        
        
        
        real_train_data_matrix = (self.train_data[index][:,np.newaxis]).T
        
        
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
        labels_tmp[self.train_labels[index]][0] = 0.99
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
        delta_Wo = to_Wo 
        delta_Wh = (to_Wh_1 * np.repeat(to_Wh_2.T, to_Wh_1.shape[1], axis=1))
        Wh = Wh - self.LR * delta_Wh
        Wo = Wo - self.LR * delta_Wo
        # print(f"Index:{index}\nenc_Wh:{Wh}\nenc_Wo{Wo}")
        # return index, to_Wo, to_Wh_1, to_Wh_2
        
        masked_Wh = self.Gama * Wh
        masked_Wo = self.Gama * Wo
        
        sig = self.gen_gradients_signature(index, (Wh, Wo), (masked_Wh, masked_Wo))
        return Wh, Wo, sig
    
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
                # print("Received parameters from server:")
                
                # print("Start to forward prop")
                result = self.forward_prop(parameters)
                # print("Finish forward prop")
                # B = self.generate_random_matrix()
                # C = self.add_matrices(A, B)
                self.send_parameters(sock, result)

# if __name__ == "__main__":
def worker(sigmap):
    HOST, PORT = "localhost", 9999
    es = ES(HOST, PORT, sigmap)
    es.run()

if __name__ == '__main__':
    processes = []
    with Manager() as manager:
        SIGMAP = manager.dict()  # Create a managed dictionary
        for i in range(num_of_client):
            # Start a new process to modify the managed dictionary
            p1 = Process(target=worker, args=(SIGMAP,))
            p1.start()
            processes.append(p1)
        for p in processes:
            p.join()

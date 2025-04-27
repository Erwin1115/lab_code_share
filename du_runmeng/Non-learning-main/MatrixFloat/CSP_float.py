import socketserver
import threading
import pickle
import struct
import numpy as np
# from MyMatrix import MyMatrix,load_matrix
import random
import threading
from utils_float import sigmoid, R, num_of_client, Precision
from utils_float import Gama, CSP_PA, Gama_
from utils_float import epochs, batch_size

from DataShare_float import load_data
from MyMatrix_float import load_matrix_float
from tqdm import tqdm
import time

from multiprocessing import Pool
compute_pool = Pool(num_of_client)

# from utils_float import compute_pool
from process_pool import calculate_parameters

print("Initializing server ...")

N = num_of_client  # Number of clients we expect to connect
LR = 0.01
# Use a threading barrier to ensure that we only proceed when all clients have sent data.
barrier = threading.Barrier(N)

Z = 128
C = 10
Wh_dim = (Z, 784)
Wo_dim = (C, Z)

Parameters_lock = threading.Lock()
Wh = np.array(((np.random.rand(*(Wh_dim))) - 0.5) * 0.25)
Wo = np.array(((np.random.rand(*(Wo_dim))) - 0.5) * 0.25)

RESULT = {}

print("Server Loadding data ...")
train_data,train_labels,test_data,test_labels = load_data()
A_Matrix, A_Matrix_ = load_matrix_float()
print("Server Loading finished")


Cursor_lock = threading.Lock()
Data_cursor = 0
cur_R = 0

compute_acc_lock = threading.Lock()

start_time = time.time()


recv_size_lock = threading.Lock()
recieved_size = []




def save_result():
    with open("result/base_line_minist", "wb") as f:
        pickle.dump(RESULT, f)


def pailliar_dec_(x):
    return CSP_PA.decrypt_ciper(int(x))
pailliar_dec = np.vectorize(pailliar_dec_)

def predict(X):
    global Wh, Wo
    
    # Remember the CSP does not have Gama. We just use it here to test the precision.
    tmp_wh = Wh * Gama_
    tmp_Wo = Wo * Gama_
    
    
    O1 = tmp_wh @ X.T
    h = sigmoid(O1)
    O = tmp_Wo @ h
    # O = sigmoid(O)
    
    
    # m = X.shape[0]
    # # 将矩阵X,y转换为numpy型矩阵
    # X = np.matrix(X)
    # # 前向传播 计算各层输出
    # # 隐层输入 shape=m*hidden_size
    # h_in = np.matmul(X, Wh.T)
    # # # 隐层输出 shape=m*hidden_size
    # h_out = sigmoid(h_in)
    # # # 输出层的输入 shape=m*output_size
    # o_in = np.matmul(h_out, Wo.T)
    # # # 输出层的输出 shape=m*output_size
    # # o_out = np.argmax(sigmoid(o_in), axis=1)
    # o_out = np.argmax(O.T, axis=1)
    o_out = np.argmax(O.T, axis=1)
    # o_out = o_in
    return o_out

def computeAcc(X, y):
    y_hat = predict(X)
    acc = np.mean(y_hat == y)
    loss =  np.mean((y - y_hat) ** 2)
    return acc,loss

    
def cost(prediction, labels):
    return np.mean(np.power(prediction - labels,2))

def compute_commutication_size():
    recv_size_lock.acquire()
    try:
        global recieved_size
        b=[915396]*len(recieved_size)
        result = 0
        for i in range(0,len(recieved_size)):result+=recieved_size[i]+b[i]
        print(f"{result} bytes, {(result)//1024} KB, {result//(1024*1024)} MB")
        recieved_size = []
    finally:
        recv_size_lock.release()

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def recv_all(self, sock, n):
        # Helper function to recv n bytes or return None if EOF is hit
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data
    
    def send_parameters(self, request, cur_cursor):
        global cur_R, Data_cursor
        # gama = Gamas[cur_R]
        
        # index, data, label = cur_data
        
        # TODO: Determine the precision
        to_1 = cur_cursor
        # print("Hiding Wh ...")
        to_2 = Gama * (Wh @ A_Matrix_[cur_cursor])
        to_3 = Gama * Wo
        to_send = (to_1, to_2.tolist(), to_3.tolist())
        # print(f"Sending {cur_cursor}...")
        # print("Ready to send")
        
        to_send_data = pickle.dumps(to_send)
        request.sendall(struct.pack('!I', len(to_send_data)))
        request.sendall(to_send_data)
        # print(f"Send size: {len(to_send_data)} bytes")

    def receive_parameters(self, request):
        response_size_data = self.recv_all(request, 4)
        if response_size_data is None:
            print("Client closed connection")
            return None
        response_size = struct.unpack('!I', response_size_data)[0]
        response_data = self.recv_all(request, response_size)
        if response_data is None:
            print("Client closed connection")
            return None
        received_parameters = pickle.loads(response_data)
        
        recv_size_lock.acquire()
        recieved_size.append(len(response_data))
        # print(f"Recieved Size: {len(response_data)} Bytes")
        recv_size_lock.release()
        
        return received_parameters

    def update_parameters(self, received_parameters):
        global Wh, Wo
        index, to_Wo, to_Wh_1, to_Wh_2 = received_parameters
        # print("Updating Parameters")
        # print(f"Received {index}")
        
        # # de_to_Wo = (pailliar_dec(to_Wo)) / (Precision ** 2)
        # # de_to_Wh1 = (to_Wh_1 @ A_Matrix_[index].T) / (Precision)
        # # de_to_Wh2 =  pailliar_dec(to_Wh_2)
        # de_to_Wo = (CSP_PA.decryMatrix(to_Wo)) / (Precision ** 2)
        # de_to_Wh1 = (to_Wh_1 @ A_Matrix_[index].T)
        # de_to_Wh2 =  CSP_PA.decryMatrix(to_Wh_2) / (Precision ** 2)
        
        # delta_Wo = de_to_Wo / N
        # delta_Wh = (de_to_Wh1 * np.repeat(de_to_Wh2.T, de_to_Wh1.shape[1], axis=1)) / N
        # # delta_Wh = ((to_Wh_1.T @ to_Wh_2).T) / N
        
        result = compute_pool.apply_async(calculate_parameters, (received_parameters,A_Matrix_[index]))  # evaluate "f(10)" asynchronously
        delta_Wh, delta_Wo = result.get(timeout=10) 
         
        Parameters_lock.acquire()
        Wh = Wh - LR * delta_Wh
        Wo = Wo - LR * delta_Wo
        Parameters_lock.release()
        
        return 
    
    
    def handle(self):
        global cur_R, Cursor_lock, Data_cursor, test_data, test_labels, compute_acc_lock, start_time
        # for j in tqdm(range(R)):
        for j in range(R):
            
            # self.send_matrix(self.request, A if j == 0 else T/N)  # Send A in the first round, else send T/N
            Cursor_lock.acquire()
            Data_cursor += 1
            cur_cursor = Data_cursor # Send A in the first round, else send T/N
            Cursor_lock.release()
            self.send_parameters(self.request, cur_cursor%batch_size)
            
            parameters = self.receive_parameters(self.request)
            if parameters is None:
                print("Client closed connection")
                return
            # T = self.add_matrices(T, C) if T is not None else C
            self.update_parameters(parameters)
            barrier.wait()
            
            # if j%100==0 and compute_acc_lock.acquire(False):
            if ((j+1)*N)%batch_size==0 and compute_acc_lock.acquire(False):
            # if compute_acc_lock.acquire(False):
                try:
                    acc,loss = computeAcc(test_data, test_labels)
                    used_time = time.time()-start_time
                    RESULT[j] = (acc, loss, int(used_time))
                    # print(f"-------Round {j} finished. Now accuracy: {acc} Loss:{loss} UsedTime:{int(used_time)}s-------")
                    if ((j+1)*N)%batch_size == 0:
                        print(f"-------Epoch {((j+1)*N)//batch_size} finished. Now accuracy: {acc} Loss:{loss} UsedTime:{int(used_time)}s-------")
                        compute_commutication_size()
                # print(f"-------Round {j} finished.-------")
                finally:
                    # Always release the lock when done
                    compute_acc_lock.release()
        
        shutdown_thread = threading.Thread(target=self.server.shutdown)
        shutdown_thread.start()
        # self.server.handler_semaphore.release()

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    # pass
class CSP:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        pass
    def run(self):
        server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)

        with server:
            ip, port = server.server_address
            print("Server running on {}:{}".format(ip, port))
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
        
            # while(not ROUND_END):
            #     time.sleep(1)
            
            server_thread.join()

if __name__ == "__main__":
    from process_pool import calculate_parameters
    HOST, PORT = "0.0.0.0", 9999
    csp = CSP(HOST, PORT)
    csp.run()
    save_result()
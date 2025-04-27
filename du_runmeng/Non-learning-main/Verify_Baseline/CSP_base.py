import socketserver
import threading
import pickle
import struct
import numpy as np
# from MyMatrix import MyMatrix,load_matrix
import random
import threading
from utils_base import sigmoid, R, baseline_reuslt_path
from utils_base import batch_size, num_of_client,malicious_rate
from DataShare_base import load_data
from tqdm import tqdm
import time


from Cryptodome.Util.number import inverse
from verify_util import *



print("Initializing server ...")

N = num_of_client  # Number of clients we expect to connect
# Use a threading barrier to ensure that we only proceed when all clients have sent data.
barrier = threading.Barrier(N)

Z = 128
C = 10
Wh_dim = (Z, 784)
Wo_dim = (C, Z)

Update_lock = threading.Lock()
Wh = np.array( ((np.random.rand(*(Wh_dim))) - 0.5) * 0.25)
Wo = np.array( ((np.random.rand(*(Wo_dim))) - 0.5) * 0.25)

Par_lock = threading.Lock()
recv_Wh_list = []
recv_Wo_list = []

time_lock = threading.Lock()
total_verify_time = 0

# result_Wh = np.zeros(Wh_dim)
# result_Wo = np.zeros(Wo_dim)

# running_ threading.Semaphore()


Aggr_lock = threading.Lock()

RESULT = {}

print("Server Loadding data ...")
train_data,train_labels,test_data,test_labels = load_data()
print("Server Loading finished")


Cursor_lock = threading.Lock()
Data_cursor = 0
cur_R = 0

recv_size_lock = threading.Lock()
recieved_size = []

compute_acc_lock = threading.Lock()
start_time = time.time()

def save_result():
    with open(baseline_reuslt_path, "wb") as f:
        pickle.dump(RESULT, f)


def predict(X):
    global Wh, Wo
    
    
    O1 = Wh @ X.T
    h = sigmoid(O1)
    O = Wo @ h
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
        print(f"{result} bytes, {(result)//1024} KB, {result//(1024*1024)} MB, {result//(1024*1024*1024)} GB")
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
        # to_1 = index
        # print("Hiding Wh ...")
        # to_2 = Wh * A_Matrix_[index]
        # to_3 = Wo
        to_send = (cur_cursor, Wh, Wo)
        # print(f"Sending {cur_cursor}...")
        # print("Ready to send")
        
        to_send_data = pickle.dumps(to_send)
        request.sendall(struct.pack('!I', len(to_send_data)))
        request.sendall(to_send_data)

    def mask_sig_is_vlaid(self, BETA, masked_gradients: np.ndarray, sig):
        prime = DiffieHellman()._prime
        sig_pk, hi, zi, R, Lambda = sig
        # if info_id!=sig_id or info_pk!=sig_pk: logging.error(f"Infoid:{info_id} SigId:{sig_id} id and pk not matched")
        
        
        # masked_wi = compress(masked_gradient, BETA)
        masked_Wh, masked_Wo = masked_gradients
        fmasked_wi = flatten(masked_Wh, masked_Wo)
        masked_wi = compress(fmasked_wi, BETA)
        left = pow(2, zi, prime)
        r_R = inverse(R, prime)
        right = (pow(2, masked_wi, prime) * r_R) % prime
        right = (pow(right, hi, prime) * int.from_bytes(sig_pk,"big")) % prime
        
        if left == right:
            # logging.info(f"User{info_id}'s gradient is valid")
            # MaskingRequestHandler.sig_map[info_id] = sig
            return True
        else:
            # logging.warn(f"User{info_id}'s gradient is INVALID")
            assert 0 == 1
            return False
    
    
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
        global Wh, Wo, total_verify_time
        recv_Wh, recv_Wo, sig = received_parameters
        i_pk, hi, zi, R, Lambda = sig
        start_time = time.time()
        if self.mask_sig_is_vlaid(BETA, (recv_Wh, recv_Wo), sig):
            Par_lock.acquire()
            # Wh = Wh + recv_Wh
            if(len(recv_Wh_list) < int(num_of_client * (1 - malicious_rate))):
                recv_Wh_list.append(recv_Wh)
                recv_Wo_list.append(recv_Wo)
            # Wo = Wo + recv_Wo
            Par_lock.release()
        end_time = time.time()
        
        time_lock.acquire()
        total_verify_time += (end_time - start_time)
        time_lock.release()
        # print(f"CSP verify cost {end_time - start_time}s")
        # print(f"Recieved {index}")
        
        # delta_Wo = to_Wo / N
        # delta_Wh = (to_Wh_1 * np.repeat(to_Wh_2.T, to_Wh_1.shape[1], axis=1)) / N
        # # delta_Wh = ((to_Wh_1.T @ to_Wh_2).T) / N
        
        # Wh = Wh - LR * delta_Wh
        # Wo = Wo - LR * delta_Wo
        
        return 
    
    
    
    def handle(self):
        global cur_R, Cursor_lock, Data_cursor, test_data, test_labels, Wh, Wo, recv_Wh_list, recv_Wo_list, total_verify_time
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
            
            
            if Update_lock.acquire(False):
                assert len(recv_Wh_list) == len(recv_Wo_list)
                result_Wh = np.zeros(Wh_dim)
                result_Wo = np.zeros(Wo_dim)   
                # print(f"Prepare to aggregate {len(recv_Wo_list)} clients.")       
                for i in range(len(recv_Wh_list)):
                    result_Wh += recv_Wh_list[i]
                    result_Wo += recv_Wo_list[i]
                Wh = result_Wh/len(recv_Wh_list)
                Wo = result_Wo/len(recv_Wh_list)
                recv_Wh_list = []
                recv_Wo_list = []
                Update_lock.release()
            barrier.wait()
            
            
            time_lock.acquire()
            if total_verify_time != 0:
                # print(f"Verify {num_of_client} RE costs {total_verify_time} seconds")
                total_verify_time = 0
            time_lock.release()
            # if time_lock.acquire(False):
            
            # print(f"-------Round {j} finished. -------")
            # if j%(batch_size//(num_of_client*10))==0 and compute_acc_lock.acquire(False):
            if (j*N)%batch_size == 0 and compute_acc_lock.acquire(False):
                try:
                    acc,loss = computeAcc(test_data, test_labels)
                    used_time = time.time()-start_time
                    RESULT[j] = (acc, loss, int(used_time))
                    # print(f"-------Round {j} finished. Now accuracy: {acc} Loss:{loss} UsedTime:{int(used_time)}s-------")
                    if (j*N)%batch_size == 0:
                        print(f"-------Epoch {(j*N)//batch_size} finished. Now accuracy: {acc} Loss:{loss} UsedTime:{used_time}s-------")
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
    HOST, PORT = "localhost", 9999
    csp = CSP(HOST, PORT)
    csp.run()
    # save_result()
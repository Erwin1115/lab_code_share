import socketserver
import threading
import pickle
import struct
import numpy as np
from MyMatrix import MyMatrix,load_matrix
import random
import threading
from DataShare import load_enc_data


print("Initializing server ...")

N = 2  # Number of clients we expect to connect
R = 2  # Number of rounds to run
cur_R = 0

# Use a threading barrier to ensure that we only proceed when all clients have sent data.
barrier = threading.Barrier(N)

Z = 1000
C = 10
Wh_dim = (Z, 784)
Wo_dim = (C, Z)
Wh = MyMatrix(matrix = np.random.randint(low= -100 ,high = 100, size=Wh_dim))
Wo = MyMatrix(matrix = np.random.randint(low= -100 ,high = 100, size=Wo_dim))

Cursor_lock = threading.Lock()
Data_cursor = 0

print("Loadding enc_data ...")
Enc_data = load_enc_data()

print("Loadding A_Matrix ...")
A_Matrix, A_Matrix_ = load_matrix()



A = np.random.rand(3,3)  # Initial matrix
T = np.zeros_like(A)  # Average matrix


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
    
    def send_parameters(self, request, enc_data):
        global cur_R, Gamas, Data_cursor, A_Matrix_
        # gama = Gamas[cur_R]
        
        index, enc_matrix, enc_label = enc_data
        
        # TODO: Determine the precision
        to_1 = index
        print("Hiding Wh ...")
        to_2 = Wh * A_Matrix_[index]
        to_3 = Wo
        to_send = (to_1, to_2, to_3)
        print("Ready to send")
        
        to_send_data = pickle.dumps(to_send)
        request.sendall(struct.pack('!I', len(to_send_data)))
        request.sendall(to_send_data)

    def receive_matrix(self, request):
        response_size_data = self.recv_all(request, 4)
        if response_size_data is None:
            print("Client closed connection")
            return None
        response_size = struct.unpack('!I', response_size_data)[0]
        response_data = self.recv_all(request, response_size)
        if response_data is None:
            print("Client closed connection")
            return None
        received_matrix = pickle.loads(response_data)
        return received_matrix

    def add_matrices(self, matrix1, matrix2):
        return matrix1 + matrix2
    
    
    
    def handle(self):
        global A, T, cur_R, Cursor_lock, Data_cursor
        for j in range(R):
            
            # self.send_matrix(self.request, A if j == 0 else T/N)  # Send A in the first round, else send T/N
            Cursor_lock.acquire()
            
            Data_cursor += 1
            cur_enc_data = Enc_data[Data_cursor] # Send A in the first round, else send T/N
            
            Cursor_lock.release()
            self.send_parameters(self.request, cur_enc_data)
            
            C = self.receive_matrix(self.request)
            if C is None:
                print("Client closed connection")
                return
            T = self.add_matrices(T, C) if T is not None else C
            barrier.wait()
            if j < R - 1:
                T = np.zeros_like(A) 
                

class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass
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
            server_thread.join()

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    csp = CSP(HOST, PORT)
    csp.run()
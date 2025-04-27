import socket
import pickle
import struct
import numpy as np

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
        index, enc_Wh, enc_Wo = parameters
        print(f"Index:{index}\nenc_Wh:{enc_Wh}\nenc_Wo{enc_Wo}")
        return
    
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
                print("Received parameters from server: \n")
                
                result = self.forward_prop(parameters)
                # B = self.generate_random_matrix()
                # C = self.add_matrices(A, B)
                self.send_parameters(sock, result)

if __name__ == "__main__":
    HOST, PORT = "localhost", 9999
    R = 2  # Number of rounds to run
    es = ES(HOST, PORT)
    es.run()
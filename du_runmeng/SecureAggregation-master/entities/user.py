import sys
import pickle
import random
import socket
import logging
import numpy as np

from utils import *

# compatible with Windows
socket.SO_REUSEPORT = socket.SO_REUSEADDR


class User:
    def __init__(self, id: str, pub_key: bytes, priv_key: bytes, precision: int):
        self.id = id
        self.host = socket.gethostname()
        self.port = int("1" + id.zfill(4))

        self.pub_key = pub_key
        self.__priv_key = priv_key
        self.pub_key_map = []

        # Used for generating session encryption key between clients.
        self.c_pk = None
        self.__c_sk = None
        
        # Used for generating mask between two clients.
        self.s_pk = None
        self.__s_sk = None
        
        #Used for generating schonar-like signature of clients.
        self.i_pk = None
        self.__i_sk = None

        self.ka_pub_keys_map = None

        self.__random_seed = None

        self.ciphertexts = None

        self.U_3 = None

        self.prec = precision
        
    def gen_DH_pairs(self):
        self.c_pk, self.__c_sk = KA.gen()
        self.s_pk, self.__s_sk = KA.gen()
        self.i_pk, self.__i_sk = KA.gen()

    def gen_signature(self):
        msg = pickle.dumps([self.c_pk, self.s_pk])
        signature = SIG.sign(msg, self.__priv_key)

        return signature

    def ver_signature(self) -> bool:
        status = True
        for key, value in self.ka_pub_keys_map.items():
            msg = pickle.dumps([value["c_pk"], value["s_pk"]])

            res = SIG.verify(msg, value["signature"], self.pub_key_map[key])

            if res is False:
                status = False
                logging.error("user {}'s signature is wrong!".format(key))

        return status

    def send(self, msg: bytes, host: str, port: int):
        """Sends message to host:port.

        Args:
            msg (bytes): the message to be sent.
            host (str): the target host.
            port (int): the target port.
        """

        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.connect((host, port))

        SocketUtil.send_msg(sock, msg)

        sock.close()

    def listen_broadcast(self, port: int):
        """Listens to the server's broadcast, and saves all users' key pairs and corresponding signatures.

        Args:
            port (int): the port used to broadcast the message.
        """

        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # # reuse port so we will be able to run multiple clients on single (host, port).
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        # # enable broadcasting mode
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # sock.bind(("", port))

        data = SocketUtil.recv_broadcast(port)

        self.ka_pub_keys_map = pickle.loads(data)

        logging.info("received all signatures from the server")

        # sock.close()

    def gen_shares(self, U_1: list, t: int, host: str, port: int):
        """Generates random seed for a PRG, generates t-out-of-U1 shares of the s_sk and random seed,
           and encrypts these shares using the shared key of the two users.

        Args:
            U_1 (list): all users who have sent DH key pairs.
            t (int): the threshold value of secret sharing scheme.
            host (str): the server's host.
            port (int): the server's port used to receive these shares.
        """

        # generates a random integer from 0 to 2**32 - 1 (to be used as a seed for PRG)
        self.__random_seed = random.randint(0, 2**32 - 1)

        n = len(U_1)

        s_sk_shares = SS.share(self.__s_sk, t, n)
        random_seed_shares = SS.share(self.__random_seed, t, n)

        all_ciphertexts = {}       # {id: ciphertext}

        for i, v in enumerate(U_1):
            if v == self.id:
                continue

            info = pickle.dumps([self.id, v, s_sk_shares[i], random_seed_shares[i]])

            v_c_pk = self.ka_pub_keys_map[v]["c_pk"]
            shared_key = KA.agree(self.__c_sk, v_c_pk)

            ciphertext = AE.encrypt(shared_key, shared_key, info)

            all_ciphertexts[v] = ciphertext

        msg = pickle.dumps([self.id, all_ciphertexts])

        # send all shares of the s_sk and random seed to the server
        self.send(msg, host, port)

    def listen_ciphertexts(self):
        """Listens to the server for the ciphertexts.
        """

        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        sock.bind(("", self.port))
        sock.listen()

        conn, _ = sock.accept()

        data = SocketUtil.recv_msg(conn)
        self.ciphertexts = pickle.loads(data)

        logging.info("received ciphertext from the server")

        sock.close()
    
    def gen_gradients_signature(self, gradients:np.ndarray, masked_gredients:np.ndarray, BETA:np.ndarray):
        
        prime = DiffieHellman()._prime
        
        
        wi, masked_wi = compress(gradients, BETA), compress(masked_gredients, BETA)
        
        
        # Pars: wi, ipki
        h = SHA256.new()
        h.update(bytes(gradients.flatten()) + self.i_pk)
        hi = int.from_bytes(h.digest(), "big")
        # Pars: hi, wi, iski
        zi = hi * wi + int.from_bytes(self.__i_sk, "big")
        R = pow(2, masked_wi - wi, prime)
        Lambda = pow(2, wi, prime)
        
        sig = (self.id, self.i_pk, hi, zi, R, Lambda)
        return sig
    
    
    def mask_gradients(self, i_gradients: np.ndarray, BETA:np.ndarray, host: str, port: int):
        """Masks user's own gradients and generates corresponding verification gradients. Then, sends them to the server.

        Args:
            gradients (np.ndarray): user's raw gradients.
            host (str): the server's host.
            port (int): the server's port used to receive the masked gradients.
        """

        logging.info(f"User{self.id}'s gradients:{i_gradients}")
        gradients = (i_gradients * self.prec).astype(int)
        U_2 = list(self.ciphertexts.keys())
        

        # generate user's own private mask vector p_u_0 and p_u_1
        rs = np.random.RandomState(self.__random_seed | 0)
        # priv_mask_vec_0 = rs.random(gradients.shape)
        priv_mask_vec_0 = rs.randint(self.prec//10, self.prec, size = gradients.shape)
        rs = np.random.RandomState(self.__random_seed | 1)
        # priv_mask_vec_1 = rs.random(gradients.shape)
        priv_mask_vec_1 = rs.randint(self.prec//10, self.prec, size = gradients.shape)

        # generate random vectors p_u_v_0 and p_u_v_1 for each user
        random_vec_0_list = []
        random_vec_1_list = []
        alpha = 0
        for v in U_2:
            if v == self.id:
                continue

            v_s_pk = self.ka_pub_keys_map[v]["s_pk"]
            shared_key = KA.agree(self.__s_sk, v_s_pk)

            random.seed(shared_key)
            s_u_v = random.randint(0, 2**32 - 1)
            alpha = (alpha + s_u_v) % (2 ** 32)

            # expand s_u_v into two random vectors
            rs = np.random.RandomState(s_u_v | 0)
            # p_u_v_0 = rs.random(gradients.shape)
            p_u_v_0 = rs.randint(self.prec//10, self.prec, size = gradients.shape)
            rs = np.random.RandomState(s_u_v | 1)
            # p_u_v_1 = rs.random(gradients.shape)
            p_u_v_1 = rs.randint(self.prec//10, self.prec, size = gradients.shape)
            if int(self.id) > int(v):
                random_vec_0_list.append(p_u_v_0)
                random_vec_1_list.append(p_u_v_1)
            else:
                random_vec_0_list.append(-p_u_v_0)
                random_vec_1_list.append(-p_u_v_1)

        # expand α into two random vectors
        alpha = 10000
        rs = np.random.RandomState(alpha | 0)
        # self.__a = rs.random(gradients.shape)
        self.__a = rs.randint(self.prec//10, self.prec, size = gradients.shape)
        rs = np.random.RandomState(alpha | 1)
        # self.__b = rs.random(gradients.shape)
        self.__b = rs.randint(self.prec//10, self.prec, size = gradients.shape)

        verification_code = self.__a * gradients + self.__b

        masked_gradients = gradients + priv_mask_vec_0 + np.sum(np.array(random_vec_0_list), axis=0)
        verification_gradients = verification_code + priv_mask_vec_1 + np.sum(np.array(random_vec_1_list), axis=0)
        
        sig = self.gen_gradients_signature(gradients=gradients, masked_gredients=masked_gradients, BETA=BETA)
        msg = pickle.dumps([self.id, masked_gradients, verification_gradients, sig])

        # send the masked gradients to the server
        self.send(msg, host, port)

    def consistency_check(self, host: str, port: int, status_list: list):
        sock = socket.socket()
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        sock.bind(("", self.port))
        sock.listen()

        conn, _ = sock.accept()

        data = SocketUtil.recv_msg(conn)
        self.U_3 = pickle.loads(data)

        logging.info("received U_3 from the server")

        signature = SIG.sign(data, self.__priv_key)
        msg = pickle.dumps([self.id, signature])

        self.send(msg, host, port)

        conn, _ = sock.accept()
        data = SocketUtil.recv_msg(conn)
        signature_map = pickle.loads(data)

        for key, value in signature_map.items():
            res = SIG.verify(pickle.dumps(self.U_3), value, self.pub_key_map[key])

            if res is False:
                logging.error("user {}'s signature is wrong!".format(key))
                status_list.append(False)

                sys.exit(1)

        sock.close()

    def unmask_gradients(self, host: str, port: str):
        """Sends the shares of offline users' private key and online users' random seed to the server.

        Args:
            host (str): the server's host.
            port (str): the server's port used to receive the shares.
        """
        
        # logging.info(f"User{self.id} prepare to unmask gradients.")

        U_2 = list(self.ciphertexts.keys())

        self.priv_key_shares_map = {}
        self.random_seed_shares_map = {}

        for v in U_2:
            if self.id == v:
                continue

            v_c_pk = self.ka_pub_keys_map[v]["c_pk"]
            shared_key = KA.agree(self.__c_sk, v_c_pk)

            info = pickle.loads(AE.decrypt(shared_key, shared_key, self.ciphertexts[v]))

            if v not in self.U_3:
                # send the shares of s_sk to the server
                self.priv_key_shares_map[v] = info[2]
            else:
                # send the shares of random seed to the server
                self.random_seed_shares_map[v] = info[3]
    
    # def share_private_random_share(self, host:str, port:str): 
        self.enc_random_seed_shares_map = {}
        # key_from:M means a client C will share the share the M's secret to key_to
        for key_from in self.random_seed_shares_map:
            for key_to in self.random_seed_shares_map:
                if key_to == key_from:
                    continue
                if key_from not in self.enc_random_seed_shares_map.keys():
                    self.enc_random_seed_shares_map[key_from] = {}
                # if key_to not in enc_random_seed_shares_map[key_from]:
                
                v_c_pk = self.ka_pub_keys_map[key_to]["c_pk"]
                shared_key = KA.agree(self.__c_sk, v_c_pk)
                to_share_rs = pickle.dumps(self.random_seed_shares_map[key_from])
                self.enc_random_seed_shares_map[key_from][key_to] = AE.encrypt(shared_key, shared_key, to_share_rs)
        
        # enc_random_seed_shares_map = random_seed_shares_map
        # Send priv_key_share and encrypted random_seed_share to server.
        msg = pickle.dumps([self.id, self.priv_key_shares_map, self.enc_random_seed_shares_map])
        # logging.info(f"User{self.id} prepare to send gradients.")
        self.send(msg, host, port)
        # logging.info(f"User{self.id} has sent gradients.")

    def listen_shared_b(self, BETA: np.ndarray, from_list: list, port:str, host:str, finish_port:str):
        
        """Listens to the server's broadcast, and saves all outputs and .

        Args:
            port (int): the port used to broadcast the message.
        """
        # 1. reveive the random_seed_share_map and masked gradients.

        # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # # reuse port so we will be able to run multiple clients on single (host, port).
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        # # enable broadcasting mode
        # sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        # sock.bind(("", port))

        data = SocketUtil.recv_broadcast(port)

        self.output, self.enc_random_seed_map, self.sig_map = pickle.loads(data)

        logging.info(f"User{self.id} received outputs and shared b from the server")

        # sock.close()
        
    # def unmask_shared_b(self, host:str, finish_port:str):
        # 2. Decrypt the random_seed_sahre_map
        secrets = {}
        for from_id in from_list:
            if self.id == from_id:continue
            my_share = self.enc_random_seed_map[(self.id, from_id)]
        
            v_c_pk = self.ka_pub_keys_map[from_id]["c_pk"]
            shared_key = KA.agree(self.__c_sk, v_c_pk)
    
            for key,value in my_share.items():
                if key not in secrets:
                    t_v_c_pk = self.ka_pub_keys_map[key]["c_pk"]
                    t_shared_key = KA.agree(self.__c_sk, t_v_c_pk)
                    info = pickle.loads(AE.decrypt(t_shared_key, t_shared_key, self.ciphertexts[key]))
                    secrets[key] = [info[3]]
                secrets[key].append(pickle.loads(AE.decrypt(shared_key, shared_key, value)))
        
        # calculate the final result.
        # logging.info(f"User{self.id} before unmask:{self.output}")
        secrets_seeds = [self.__random_seed]
        for s in secrets:
            secrets_seeds.append(SS.recon(secrets[s]))
        for random_seed in secrets_seeds:
            rs = np.random.RandomState(random_seed | 0)
            # priv_mask_vec_0 = rs.random(self.output.shape)
            priv_mask_vec_0 = rs.randint(self.prec//10, self.prec, size = self.output.shape)
            self.output -= priv_mask_vec_0
        
        # Verify the sum from server
        to_check = random.choice(list(self.sig_map.keys()))
        sig_id, sig_pk, hi, zi, R, Lambda = self.sig_map[to_check]
        
        prime = DiffieHellman()._prime
        wi = compress(self.output, BETA)
        left = pow(2, wi, prime)
        left = (left * pow(Lambda, hi-1, prime)) % prime
        left = (left * int.from_bytes(self.ka_pub_keys_map[to_check]["i_pk"], "big")) % prime
        
        right = pow(2, zi, prime)
        for o_u in self.sig_map:
            if o_u != to_check:
                _, _, _, _, _, tmp_Lambda = self.sig_map[o_u]
                right = (right * tmp_Lambda) % prime
        
        self.output = self.output / self.prec
        if left == right:
            logging.info(f"User{self.id} finish verifying server's aggregation, unmask result:{self.output}")
            self.send(pickle.dumps(self.id), host, finish_port)
        else:
            logging.warning(f"User{self.id} receive forged result:{self.output}")
            # logging.warning(f"User{self.id} Left:{left}----------Right:{right}")
            logging.warning(f"User{self.id} Left-Right:{left-right}")
            self.send(pickle.dumps(self.id), host, finish_port)
        return
    
    def verify(self, output_gradients, verification_gradients, num_U_3):
        gradients_prime = self.__a * output_gradients + num_U_3 * self.__b

        return ((gradients_prime - verification_gradients) < np.full(output_gradients.shape, 1e-6)).all()

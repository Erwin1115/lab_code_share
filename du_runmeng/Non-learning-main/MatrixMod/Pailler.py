import os
from gmpy2 import invert
from phe import paillier, EncryptedNumber, encoding,PaillierPublicKey, PaillierPrivateKey
import numpy as np
from functools import reduce
import time
import sys

class PaillierBase(object):
    def __init__(self):
        pass

    def generate_key(self):
        # self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=1024)
        # print("父类公钥： ", self.public_key)
        # 1024 bits
        n = 105211277601329764885976371553931740693942552614506943030741436633367438038838483661538259535207494737477671983506426710625582100407229174445903162834110742727477567820553270304259281760392205725364989886168177892524590609532732953140015864565410487968575587131028160387944484831400677283575900893850878519611
        p = 7907053922330812426645581944513533811027920742785042972150780423568693183201661580338199392724800288469764444402143452551906413144056364094973185367621547
        q = 13306002290460663685267506123156189923241007948539178191654309945010168295659175554070274533372523394802878392969139399501854952765056693985536855464078513
        
        self.public_key = PaillierPublicKey(n)
        self.private_key = PaillierPrivateKey(self.public_key, p, q)
        return self.public_key, self.private_key

    def public_key_encrypt(self, x):
        return self.public_key.encrypt(x)

    def generate_random(self):
        N = 4
        self.randint = int.from_bytes(os.urandom(N), byteorder="big")
        return self.randint


class PaillierCreator(PaillierBase):
    def __init__(self):
        super(PaillierCreator, self).__init__()
        self.public_key, self.private_key = self.generate_key()
        no_use1 = self.public_key.__dict__
        no_use2 = self.private_key.__dict__
        self.n = no_use1['n']
        self.g=no_use1['g']
        self.p=no_use2['p']
        self.q=no_use2['q']


    def encrypt(self,a):
        # f = open('PaillierPlustest.txt', mode='w')
        Ea_object = self.public_key.encrypt(a)
        # print("第一个加密对象对应的密文： ", self.Ea_object)  # 就是对象
        # self.Ea = self.Ea_object._EncryptedNumber__ciphertext
        # print("第一个加密对象对应的密文： ", self.Ea)  # 就是密文
        return Ea_object._EncryptedNumber__ciphertext

    def decrypt(self,Ea_object):
        # self.decrypt_object = paillier.PaillierPrivateKey(self.public_key,self.p,self.q)
        # self.a= self.decrypt_object.raw_decrypt(Ea)
        a=self.private_key.decrypt(Ea_object)
        # print("解密对应的明文",self.a)
        return a

    def decrypt_ciper(self, Ea):
        # 将Ea转化为对象
        Ea_object = EncryptedNumber(self.public_key, int(Ea))
        # print("Ea_object对象", Ea_object)
        a = self.private_key.decrypt(Ea_object)
        # print("明文", a)
        return a

    def object_to_ciper(self,Ea0):
        Ea=Ea0._EncryptedNumber__ciphertext
        # print("Ea0对象对应的密文",Ea)
        return Ea

    def ciper_to_object(self, Ea):
        Ea0 = EncryptedNumber(self.public_key, Ea)
        # print("Ea0对象对应的密文", Ea0)
        return Ea0

    # def generate_part_private(self,number):
    #     for i in range(number):
    #         self.r=random.randint(0,9)
    #         print(self.r)
    #         print(self.n)
    #         a=np.mod((self.r)**(self.n),(self.n)**2)
    #         print(a)
    def add_ciper(self,Ea,Eb):
        sum_object = Ea._add_encrypted(Eb)
        # print("密文",self.sum_object)
        # self.sum_object = EncryptedNumber(self.public_key, self.sum_ciphertext)
        # print("密文和",self.sum_object)
        return sum_object

    def add_plain(self, Ea,b):
        Ec_object=Ea._add_scalar(b)
        # print("Ea和一个明文相加对应的密文： ", self.Ec_object)  # 就是密文
        return Ec_object

    def mul_plain(self, Ea,b):
        b=int(b)
        # if type(b)="int":
        Ec=Ea._raw_mul(b)
        # print("Ea和一个明文相加对应的密文： ", self.Ec)  # 就是密文
        Ec_object = EncryptedNumber(self.public_key, Ec)
        # print("Ea和一个明文相乘对应的密文： ", self.Ec_object )  # 就是密文
        return Ec_object

    def encryMatrix(self,data):
        list_ret = []
        # print("加密数据类型： ", type(data))
        row = data.shape[0]
        col = data.shape[1]
        # print("加密入参数据类型： ", row, "  ", col)
        for i in range(row):
            for j in range(col):
                list_ret.append(self.public_key.encrypt(float(data[i][j])))
        EM=np.array(list_ret).reshape(row,col)
        # print("EM",EM)
        # print(type(EM))
        # print(EM.shape)
        return EM

    def decryMatrix(self,Edata):
        list_ret = []
        # print("加密数据类型： ", type(Edata))
        row = Edata.shape[0]
        col = Edata.shape[1]
        # print("加密入参数据类型： ", row, "  ", col)
        for i in range(row):
            for j in range(col):
                list_ret.append(self.private_key.decrypt(Edata[i][j]))
        M=np.array(list_ret).reshape(row,col)
        # print("M",M)
        # print(type(M))
        # print(M.shape)
        return M

    def addMatrix(self,Ea,Eb):
        list_ret = []
        # print("加密数据类型： ", type(Edata))
        row = Ea.shape[0]
        col = Ea.shape[1]
        # print("加密入参数据类型： ", row, "  ", col)
        for i in range(row):
            for j in range(col):
                list_ret.append(self.add_ciper(Ea[i][j],Eb[i][j]))
        Ec=np.array(list_ret).reshape(row,col)
        # print("Ec",Ec)
        # print(type(M))
        # print(M.shape)
        return Ec

    def generate_partial_key(self,number):
        list=[]
        for i in range(number):
            r=self.generate_random()
            partial_key = (r ^ self.n) % self.public_key.nsquare
            partial_key=self.ciper_to_object(partial_key )
            list.append(partial_key)
            partial_key_invert = invert(r ^ self.n, self.public_key.nsquare)
            partial_key_invert = self.ciper_to_object(partial_key_invert)
            list.append(partial_key_invert)
        partial_key_array=np.array(list).reshape(number,number)
        # print(partial_key_array)
        # print(partial_key_array.shape)
        return partial_key_array

    def recive_all_partial_key(self,number,partial_key_array):
        list_1=[]
        list_2=[]
        for i in range(number):
            list_1.append(partial_key_array[i][0])
        for i in range(number):
            list_2.append(partial_key_array[i][1])
        # 第一个参与者
        pa_all_key=np.array(list_1)
        pb_all_key=np.array(list_2)
        return pa_all_key,pb_all_key

    def martix_twice_encrypt(self,Ematrix,partial_key_array):
        list=[]
        row=Ematrix.shape[0]
        col=Ematrix.shape[1]
        for i in range(row):
            for j in range(col):
                temp=self.add_ciper(Ematrix[i][j],partial_key_array[0])
                temp1=self.add_ciper(temp,partial_key_array[1])
                list.append(temp1)
        twice_encrypt_matrix=np.array(list).reshape(row,col)
        # print("二次加密",twice_encrypt_matrix)
        return twice_encrypt_matrix


def mod_pow(base, exponent, modulus):
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent // 2
        base = (base * base) % modulus
    return result

def test(pa:PaillierCreator):
    # 参与者A生成部分私钥
    # pa = PaillierCreator()
    # r = 8 % pa.public_key.nsquare

    # print(r)
    # R = (r ^ pa.n) % pa.public_key.nsquare
    c=(
              (mod_pow(pa.g, 8, pa.public_key.nsquare)) * (mod_pow(4, pa.n, pa.public_key.nsquare))
       ) % pa.public_key.nsquare
    print("Ciphertext:", c)
    c_ob=pa.ciper_to_object(c)
    # print("mm", c_ob)
    m=pa.decrypt(c_ob)
    print("mm",m)

if __name__ == '__main__':
    # 模拟测试验证开销
    pa=PaillierCreator()
    test(pa)
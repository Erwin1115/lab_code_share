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
        
        # 1280 bits
        # n = 12290198531442815972737197949204762581093164453339523443532657633851005988091953103311367325049780254433251357429619054593809849970823766806633376591781147362829595952662349092870862044844671559660759902015576472173292191025417101498280258306741392551927225485718405993990736435197603286017215390654566998509630819884391445779541161166337601479118601665459805150011605359041854185234349
        # p = 3114375916026484125918919523629353097652741419321219320755669200323004832277650462260203168575836897383495456804913272182249090774274315619278765708680530107576324839185179697197465254005547461
        # q = 3946279724357559604120067730301402918607940338561469248004636926046969784232574797705525431711091934556741545013474781991986933664105566864313743570226116317349874058949598327115011316959915209
        
        # # 1536 bits
        # n = 1454137920699471277889478806139202280301205563267572865562138711179284266253697149216015095831334672390279956746686899461947925649947561639669093550870786259873198239651596711086296829305978731911031282242429286879181586342684948775076138855632680763387622288876380658950948080855999623047426182525017266189959181036188007253901822826696074700408837386532078974122391347515171550211217183121830532378236258878438518626574801314608686954125959856548340030978202849
        # p = 1096007438705325473876193669540316506702926561663321995347637410677410069156021217109582650442906307770641062493097968120290898376475349997535273217007638744648802975738386028689805796774908601459808656234211676525662635001660016469
        # q = 1326759170920584789809000828391629962900182111824954324544812214337843661131904132332262719505260359769090381442604940017083326307874407617746415715090260145502847730212856294761577556008475430621995733418228541529641068894105053021
        
        
        # # 1792 bits
        # n = 173563543230873330865428429828435052032117936914609450201030279401468136120988507365215824043641682718808016097619410709263352608223362526331437756429915023229063278216766801029787612276771282256987996769427498738114107136105961898895192769356533899763409247284950597547081530016435356718217382698020455579425074503418871666868540648115735485947892793271350261854679543768192168444189434980700991628064687825263397532755184671477977563466318901802452203708342060279945501411905178966817557396982777187405772274493268880972683329952023056063
        # p = 358665400177123771533914225595728759358467714992513982551516297024691652544159374094978268891451969613960301915346147062629594006965951047802943943021724599005543546466991783661828384700193114494139407147703720060585494988440132348112422567369189075659298528946094018349
        # q = 483914933375676864186915484623581039179084383697999194887290400123099548167353054352227176129775209100986396618031662984406157513986079358355259584674522900859951690453096534025156502559474134387462647984529247985845798936495811931438336796976602376471963526305525295387
        
        # # 2048 bits
        # n = 20109801385923042779043087302134218842871303282905979638343123857621628412800423626525874345722011598740185236799829009119388729304183852773917718448290591656874357941568857101261452369606249280521786511715780198096045060518336007221491228545943340809895792014156230539246050363684282132985292850393381191045939515396709148307701582332501657242341767225528280842860091138744786590386080879287570298121886564833729141861495904143852445747116331153160495681261801306344523016525696022681460213517569232978260502535859331592920193974332496402447196022236207366846311310867400399889407922855314286691656118239612943083511
        # p = 133888538655823813113156212236041034236263848727122440904468062290839928236949036551493001429449963778150893095626372181467086748357114610035009808156052397420947414323919068828088739097815464705693836847299771477241298932324433535597325265232682852464866901568038248343596223590242416677495622086147465731647
        # q = 150198079595279212747798705712250480032464889023836637917306132986488602105284224639401900535097096936734945960471455996905684913995075810104375979006268287594107249347268643555299333636396333450562419094265368997388175938136537042511032914647339394640905813086491564080864072291921762852762680508204337674313
        
        
        
        
        
        
        
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
        # return Ea_object._EncryptedNumber__ciphertext
        return Ea_object

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
    
    # def mul_ciper(self, Ea, Eb):
    #     mul_object = Ea

    def encryMatrix(self,data):
        list_ret = []
        # print("加密数据类型： ", type(data))
        row = data.shape[0]
        col = data.shape[1]
        # print("加密入参数据类型： ", row, "  ", col)
        for i in range(row):
            for j in range(col):
                # list_ret.append(self.public_key.encrypt(float(data[i][j])))
                list_ret.append(self.public_key.encrypt(int(data[i][j])))
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
    
    def mulscalarMatrix(self,Ea,b):
        list_ret = []
        # print("加密数据类型： ", type(Edata))
        row = Ea.shape[0]
        col = Ea.shape[1]
        # print("加密入参数据类型： ", row, "  ", col)
        for i in range(row):
            for j in range(col):
                list_ret.append(self.mul_plain(Ea[i][j],b))
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
    
    a = 3.22
    b = 0.28
    Ea = pa.encrypt(a)
    Eb = pa.encrypt(b)
    Ec = Ea - Eb
    print(f"{a}-{b}={pa.decrypt(Ec)}")
    
    Ed = pa.mul_plain(Eb, 10)
    Ec = Ea - Ed
    print(f"10 * {b} = {pa.decrypt(Ed)}")
    print(f"{a}-10*{b}={pa.decrypt(Ec)}")
    
    
    print("---------------------------------------------")
    a = 322
    b = 28
    Ea = pa.encrypt(a)
    Eb = pa.encrypt(b)
    Ec = Ea - Eb
    print(f"{a}-{b}={pa.decrypt(Ec)}")
    
    Ed = pa.mul_plain(Eb, 10)
    Ec = Ea - Ed
    print(f"10 * {b} = {pa.decrypt(Ed)}")
    print(f"{a}-10*{b}={pa.decrypt(Ec)}")
    
    n = 100
    matrix_plain = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(int)
    matrix_cipher = pa.encryMatrix(matrix_plain)
    matrix_cipher_mul = pa.mulscalarMatrix(matrix_cipher, n)
    matrix_cipher_mul_dec = pa.decryMatrix(matrix_cipher_mul)
    
    print(f"Origin Matrix:\n{matrix_plain}")
    print(f"Origin Matrix * {n}:\n{matrix_cipher_mul_dec}")
    print(f"Origin Matrix * {n} - Matrix:\n{pa.decryMatrix(matrix_cipher_mul-matrix_plain)}")
    
    
    MA = np.array([[1,0,0],[0,1,0],[0,0,1]]).astype(int)
    MB = np.array([[1,1,1],[1,1,1],[1,1,1]]).astype(int)
    MA_cipher = pa.encryMatrix(MA)
    # MB_cipher = pa.encryMatrix(MB)
    print(f"A:\n{MA}")
    print(f"B:\n{MB}")
    print(f"A@B:\n{pa.decryMatrix(MA_cipher @ MB)}")
    
    
    
    # while True:
    #     val = float(input("Enter any number: "))
    #     enc_obj = pa.encrypt(val)
    #     print(f"Origin Number:{val}")
    #     print(f"Encrypted Value:{pa.object_to_ciper(enc_obj)}")
    #     dec_obj = pa.decrypt(enc_obj)
    #     print(f"Decrypted Value:{dec_obj}")
    #     print("---------------------------------------------")
    
    # 参与者A生成部分私钥
    # pa = PaillierCreator()
    
    
    # r = 8 % pa.public_key.nsquare

    # print(r)
    # R = (r ^ pa.n) % pa.public_key.nsquare
    # c=(
    #           (mod_pow(pa.g, 8, pa.public_key.nsquare)) * (mod_pow(4, pa.n, pa.public_key.nsquare))
    #    ) % pa.public_key.nsquare
    # print("Ciphertext:", c)
    # c_ob=pa.ciper_to_object(c)
    # # print("mm", c_ob)
    # m=pa.decrypt(c_ob)
    # print("mm",m)

if __name__ == '__main__':
    # 模拟测试验证开销
    pa=PaillierCreator()
    test(pa)
import sys
import random
from time import time

from phe import paillier
import numpy as np
import os
import pandas as pd
# from sklearn.datasets import fetch_openml
# from keras.datasets import mnist
# from tensorflow.examples.tutorials.mnist import input_data

# 矩阵加密
def encryMatrix(data,public_key):
    list_ret = []
    print("加密数据类型： ",type(data))
    # data=data.tolist()
    label = 0
    if isinstance(data,list):
        if isinstance(data[0], np.uint8):
            label = 1
        elif isinstance(data[0], str):
            label = 2
        for i in range(0, len(data)):
            if label == 1:
                data[i] = int(data[i])
            elif label == 2:
                data[i] = float(data[i])
            list_ret.append(public_key.encrypt(data[i]))
        return list_ret
    else:
        shape = data.shape
        print("加密入参数据类型： ", shape,"  ",len(shape))
        if len(shape) ==1:
            for i in range(0, shape[0]):
                  list_ret.append(public_key.encrypt(data[i]))
            return np.array(list_ret).reshape(shape[0],1)
        else:
            label = 0
            # 因为数据集的数据类型不一样 在加密时有细微区别
            if isinstance(data[0][0], np.uint8):
                label = 1
            elif isinstance(data[0][0], str):
                label = 2
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    if label == 1:
                        data[i][j] = int(data[i][j])
                        list_ret.append(public_key.encrypt(int(data[i][j])))
                    elif label == 2:
                        data[i][j] = float(data[i][j])
                        list_ret.append(public_key.encrypt(int(data[i][j])))
                    else:
                        list_ret.append(public_key.encrypt((data[i][j])))
            return np.array(list_ret).reshape(shape)

# 矩阵解密
def decryMatrix(data,private_key):
    print("解密数据类型： ", type(data))
    if isinstance(data,list):
        for i in range(0, len(data)):
            data[i] = private_key.decrypt(data[i])
    else:
        shape = data.shape
        print("解密入参数据类型： ", shape, "  ", len(shape))
        if len(shape) == 1:
            for i in range(0, shape[0]):
                data[i] = private_key.decrypt(data[i])
        else:
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    data[i][j] = private_key.decrypt(data[i][j])
    return data

def DoctorDu(feature,iter,batchsize,A,B,dataset,data,pu,pr):
    # start = time()
    start1 = time()
    memSizeList1 = []
    # 预处理实验：一次性加密所有数据标签  假设最后一列是标签
    # 输出100个
    # print(dataset[1].tolist())
    # print(len(dataset[1].tolist()))
    lable_encry=encryMatrix(dataset[1].tolist(),pu)
    print("标签加密结束")
    # 一个加密标签的内存
    print(type(lable_encry))
    memSizeList1.append(lable_encry)
    # 一个有可逆矩阵A保护的特征矩阵
    memSizeList1.append(dataset.values.tolist()[0])
    # 一个有可逆矩阵B保护的特征矩阵
    memSizeList1.append(dataset.values.tolist()[0])
    print("算法1计算和通信结束")
    end1 = time()
    print("du预处理总时间； ", end1 - start1)
    # print("du预处理列表通信时间", memSizeList1)

    # 传输数据量
    x=iter+(iter*feature)+(iter*feature)
    A=np.random.rand(x).tolist()
    print("du预处理通信存储", sys.getsizeof(A))








def Li(feature,iter,batchsize,A,data,data1,pu,pr):
    print("batchsize",batchsize)
    start1=time()
    memSizeList1 = []
   # 加密全量特征参数，预处理实验
   # 假设加密1w条 memSizeList.append(sys.getsizeof(dataset.values[:10000]))
    feature_encry=encryMatrix(data.values,pu)

    # print("feature_encry",feature_encry)
    # 因为数组形式永远都是120 要转化为list
    feature_encry=feature_encry.tolist()

    # 加密特征值的内存
    memSizeList1.append(feature_encry)
    # 一个A的内存和A-1 内存
    memSizeList1.append(A.values.tolist())
    memSizeList1.append(A.values.tolist())
    #     # 一个y的内存
    memSizeList1.append(data[1].tolist())
    end1 = time()
    print("li算法1总时间: ", end1 - start1)

    # 传输量
    x=iter+(iter*feature)*3
    A=np.random.rand(x).tolist()
    print("li算法1通信总时间",  sys.getsizeof(A))



def Nonll(iter,feature,newclient,client,dataset,pu,pr):
    s=time()
    memSizeList1 = []
    # 预处理实验。
    start1 = time()
    for i in range(0,newclient):
        #TODO 加密(d)*d 的矩阵
        rand = dataset[:feature] #取的和feature对应一样的一个正矩阵
        print("取的正矩阵：",type(rand),"   ", rand.shape)
        # tmp = sys.getsizeof(rand)
        print("rand",rand)
        enRand = encryMatrix(rand,pu)
        # 加密的特征值上传通信
        memSizeList1.append(enRand.tolist())
        print("enRand.tolist()",memSizeList1)
    end1 = time()
    print("wang预处理总时间： ", end1 - start1)

    #传输数据量
    x = feature * feature
    A = np.random.rand(x).tolist()
    print("wang预处理通信总时间", sys.getsizeof(A))




# 读取CIFAR10的函数
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        print(dict.keys())
        # 可以看出这个字典的keys
        # print(dict.__contains__(b'data'))
    X = dict[b'data']
    # print("X的类型： ", type(X))
    # print("X的形状： ", X.shape)
    # data = pd.DataFrame(X)
    # print(data)
    # print("data的类型： ", type(data))
    # print("第一行： ", data.iloc[0],"   类型：  ", type(data.iloc[0]))
    # print("第一列： ",data[0],"   类型：  ", type(data[0]))
    return X

# 输入文件目录，输出数据集的训练集数据和标签，测试集数据和标签
def load_cifar(ROOT):
    xs = []
    ys = []
    ret = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % b)
        data= unpickle(f)
        ret.append(data)
    result = []
    for i in range(0,len(ret)-1):
        if i == 0:
            result = np.vstack((ret[i],ret[i+1]))
        else:
            result = np.vstack((result,ret[i+1]))
    return pd.DataFrame(result)

# def LoadMnist():
#     mnist_data = fetch_openml("mnist_784")
#     X = mnist_data["data"]
#     y = mnist_data["target"]

#     y = np.vstack(y)
#     print(X.shape)
#     print(y.shape)
#     data = np.hstack((X, y))
#     print(data.shape)
#     data = pd.DataFrame(data)
#     print("数据集：", data)
#     print("类型： ", type(data))
#     # data.to_csv(r'./MNIST.csv', header=None, index=None)
#     return data

if __name__ == '__main__':
    public_key, private_key = paillier.generate_paillier_keypair()
    # 如果用cifar10，就注释掉上这行代码
    # data = load_cifar('C://Users//luke//Downloads//cifar-10-python//cifar-10-batches-py')
    # 加在数据集





    # print("开始加载数据集")
    # data = LoadMnist()
    # # print("数据集", data)
    # print("加载结束")

    # # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # # data = mnist.test.images  # (10000,784)
    # # print("数据集：",data)
    # # print("类型： ", type(data))

    # # 三个变量，固定两个，变换一个看结果

    # # 客户端 100 200 300 400 500
    # client=100
    # # 样本量 1000 2000 3000 4000 5000
    # samples=200
    # # 模拟并行处理计算
    # bingxing_sample=int(samples/client)
    # # 特征 50 100 150 200 350
    # feature =20

    # # 批次 2 8 32 128
    # batchsize=8

    # # 迭代次数 500 1000 1500 2000 2500
    # iter=int(samples/client)

    # # mnist 取前6w行
    # # 这里是并行加密
    # data1 = data.head(bingxing_sample)
    # # 仅取出标签
    # # lable = data.iloc[:, 784:785]
    # # print("标签", lable)
    # # 仅仅取出特征值，取多少列
    # data1 = data1.iloc[:, 0:feature]
    # # print("特征",feature_data)
    # # 合并数据集
    # # data = pd.concat([feature_data, lable], axis=1)
    # print("取出来的数据集",data1)

    # # 取出来一个d×d的矩阵
    # A=data.iloc[0:feature,0:feature]
    # print("A",type(A)) #矩阵 数据
    # # 取出来一个n×n的矩阵
    # B=data.iloc[0: bingxing_sample,0: bingxing_sample]
    # print("B", type(B)) #矩阵 数组



    # # Li(feature,iter,batchsize,A,data1,data,public_key,private_key)
    # DoctorDu(feature,iter,batchsize,A,B,data1,data,public_key, private_key)

    # #客户端也需要并行,需要new一个新的客户端
    # new_client=int(client/client)
    # data2 = data.head(feature)
    # data2 = data2.iloc[:, 0:feature]
    # print("为客户并行取出来的数据集", data2)

    # Nonll(iter,feature,new_client,client,data2, public_key, private_key)

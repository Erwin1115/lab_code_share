import math
import random
import numpy as np
from multiprocessing.pool import ThreadPool
import time
from mxnet import nd



class power:

	def __init__(self, a, N):
		self.a = a
		self.N = N
		if isinstance(self.a, power):  # and isinstance(self.N, int):
			self.a = a.a
			self.N = a.N*N
		if isinstance(self.N, power):
			raise Exception('Not support power on top.') 
	def __str__(self):
		return '{}^{}'.format(self.a, self.N)
	def value(self):
			return self.a**self.N
#print(isinstance(1, int))
#quit()	
			
class powmultiply:

	def __init__(self, a, b):
		if isinstance(a, int):
			self.a = pow(a, 1)
		if isinstance(b, int):
			self.b = pow(b, 1)
		self.a = a
		self.b = b	
		if self.a.a == self.b.a:
			self.a = power(self.a.a, self.a.N+self.b.N)
			self.b = power(1, 1)
	def __str__(self):
		return '{}*{}'.format(self.a, self.b)
	def value(self):
		if isinstance(self.a, power) and isinstance(self.b, power):
			return self.a.value() * self.b.value()
	def to_power(self):
		if self.b.value() == 1:
			return self.a
		else:
			raise Exception('cannot transform to power')
			
class powdiv:

	def __init__(self, a, b):
		if isinstance(a, int):
			self.a = pow(a, 1)
		if isinstance(b, int):
			self.b = pow(b, 1)
		self.a = a
		self.b = b	
		if self.a.a == self.b.a:
			self.a = power(self.a.a, self.a.N-self.b.N)
			self.b = power(1, 1)

	def __str__(self):
		return '{}*{}'.format(self.a, self.b)

	def value(self):
		if isinstance(self.a, power) and isinstance(self.b, power):
			return self.a.value() / self.b.value()

	def to_power(self):
		if self.b.value() == 1:
			return self.a
		else:
			raise Exception('cannot transform to power.')
		
#print(powmultiply(power(2, 2), power(2, 3)))
#quit()

class DDHIP_Setup(object):
	def __init__(self, l, p=0b1101, g=5):
		self.p = p
		self.msk = s = [random.randint(1, p) for _ in range(l)]
		self.mpk = [power(g, si) for si in s]
	
	def setup(self):
		return self.mpk, self.msk
	

class DDHIP_Encrypt(object):

	def __init__(self, x, mpk, msk, g=5, p=0b1101):
		if len(msk) < len(x) or len(msk) != len(mpk):
			raise Exception('length not match.')
		self.l = len(x)
		self.p = p
		self.g = g
		self.x = x
		self.r = random.randint(1, p)
		self.mpk = mpk
		self.msk = msk
	
	def encrypt(self, ):
		ct = [power(self.g, self.r)] + [powmultiply(power(self.g, self.x[i]),power(self.mpk[i], self.r)).to_power() for i in range(self.l)]
		return ct
		
class DDHIP_Decrypt(object):

	def __init__(self, y, msk, ct):
		self.ct = ct
		self.sk_y = 0
		self.y = y
		self.msk = msk
		self.keyder()
	
	def keyder(self):
		for i in range(len(self.y)):
			self.sk_y += self.y[i]*self.msk[i]
			
	def decrypt(self, ):
		ct0_y = power(self.ct[0], self.sk_y)
		ct = self.ct[1:]
		_pi = power(ct[0], self.y[0])
		for i in range(1, len(ct)):
			_pi = powmultiply(_pi, power(ct[i], self.y[i])).to_power()
		return powdiv(_pi, ct0_y).to_power().N
		  
		
class UseDot:

	def __init__(self, l):
		setup = DDHIP_Setup(l=l)
		self.mpk, self.msk = setup.setup()

	def use_dot_one(self, list1, list2):
		start = time.time()
		encrypt = DDHIP_Encrypt(list1, self.mpk, self.msk)
		ct = encrypt.encrypt()
		# print('加密时间', time.time() - start)
		decrypt = DDHIP_Decrypt(list2, self.msk, ct)
		return decrypt.decrypt()







# u = UseDot(784)
# w1 = nd.random.normal(scale=1, shape=(10, 784)).asnumpy()
# start = time.time()
# ndar = nd.zeros(shape=(10, 10))
# list2 = []
# for i in range(10):
# 	start_1 = time.time()
# 	list1 = []
# 	for j in range(10):
# 		res = u.use_dot_one(w1[i, :], w1.T[:, j])
# 		list1.append(res)
# 		ndar[i, j] = res
# 	list2.append(list2)
# 	print(time.time() - start_1)
# # print(nd.array(np.array(list2)))
# print(ndar)
# print(time.time() - start)



#
# class Dot:
# 	def __init__(self, m0, m1):
# 		# if isinstance(m0, np.ndarray) and isinstance(m1, np.ndarray):
# 		self.m0 = m0
# 		self.m1 = m1
# 		self.u = UseDot()
#
# 		# else:
# 		# 	raise Exception('Non-nparray data types are not supported.')
# 		if m0.shape[1] != m1.shape[0]:
# 			raise Exception(f'shapes {m0.shape} and {m1.shape} not aligned: {m0.shape[1]} (dim 1) != {m1.shape[0]} (dim 0).')
#
# 	def dot(self):
# 		m0 = self.m0
# 		m1 = self.m1
# 		self.dim = self.m0.shape[1]
# 		# self.list1 = np.zeros(shape=(self.m0.shape[0], self.m1.shape[1]))
# 		self.list1 = []
# 		# p = [[i, j] for i in range(self.m0.shape[0]) for j in range(self.m1.shape[1])]
# 		# pool = ThreadPool(100)
# 		# pool.map(self.run, p)
# 		for i in range(self.m0.shape[0]):
# 			list2 = []
# 			for j in range(self.m1.shape[1]):
# 				list2.append(self.u.use_dot_one(self.dim, m0[i, :], m1[:, j]))
# 				print(i, ' ', self.m0.shape[0], ' ', j, ' ', self.m1.shape[1])
# 			self.list1.append(list2)
# 		return self.list1
#
#
# 	def run(self, p):
# 		i = p[0]
# 		j = p[1]
# 		self.list1[i][j] = self.u.use_dot_one(self.dim, self.m0[i, :], self.m1[:, j])
# 		# print(i, j)
#
# from mxnet import nd
#
# w1 = nd.random.normal(scale=1, shape=(1000, 1000))
# print(w1.shape)
#
# u = Dot(w1, w1.T)
#
# print(u.dot())
#


		
# The following example shows how to calculate [1, 2, 5]x[2, 4, 5] make sure l = len(vector)
# if __name__ == '__main__':
# 		u = UseDot()
# 		print(u.use_dot(5, [1.88885, 2, 5, 1.88885, 5], [2998.888, 0.44, 59990, 50000, 5]))
# 		setup = DDHIP_Setup(l=5)
# 		mpk, msk = setup.setup()
# 		encrypt = DDHIP_Encrypt([1.88885, 2, 5, 1.88885, 5], mpk, msk)
# 		ct = encrypt.encrypt()
# 		list1 = list([2998.888, 0.44, 59990, 50000, 5])
# 		decrypt = DDHIP_Decrypt(list1, msk, ct)
# 		print(decrypt.decrypt())

# quit()


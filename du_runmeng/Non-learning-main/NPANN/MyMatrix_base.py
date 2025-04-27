
import numpy as np
from copy import deepcopy
import pickle
import sympy
from utils_base import A_Matrix_path, A_Matrix_float_path

matrix_P = 262139


class ModularException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class MyMatrix:
    def __init__(self, matrix, p=matrix_P):
        if isinstance(matrix, np.ndarray):
            matrix = matrix.tolist()
        if not all(len(row) == len(matrix[0]) for row in matrix):
            raise ValueError("All rows of the matrix must have the same length")
        self.matrix = matrix
        self.shape = (len(matrix), len(matrix[0]))
        self.p = p

    def _multiplicative_inverse(self, a):
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            else:
                g, y, x = extended_gcd(b % a, a)
                return g, x - (b // a) * y, y

        g, x, _ = extended_gcd(a, self.p)
        if g != 1:
            raise ModularException('Modular inverse does not exist')
        else:
            return x % self.p

    def __mul__(self, other):
        if isinstance(other, MyMatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("The number of columns of the first matrix must equal the number of rows of the second matrix")
            result = [[0 for _ in range(other.shape[1])] for _ in range(self.shape[0])]
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        result[i][j] += self.matrix[i][k] * other.matrix[k][j]
                        result[i][j] %= self.p
            return MyMatrix(result, self.p)
        elif isinstance(other, (int, float, complex)):
            result = [[0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i][j] = (self.matrix[i][j] * other) % self.p
            return MyMatrix(result, self.p)
        else:
            raise ValueError("Invalid type for multiplication")
        
    def __rmul__(self, other):
        if isinstance(other, MyMatrix):
            raise ValueError("Matrix multiplication is not commutative, use 'MyMatrix * other' instead")
        elif isinstance(other, (int, float, complex)):
            return self.__mul__(other)
        else:
            raise ValueError("Invalid type for multiplication")

    def __str__(self):
        matrix_str = []
        for i, row in enumerate(self.matrix):
            if len(row) > 6:
                row = row[:3] + ['...'] + row[-3:]
            if i == 3 and len(self.matrix) > 6:
                matrix_str.append('...')
            if i < 3 or i >= len(self.matrix) - 3 :
                matrix_str.append(' '.join(map(str, row)))
        return "-------%s-------\n"%str(self.shape) + '\n'.join(matrix_str)+ "\n"
    
    def __eq__(self, other):
        if isinstance(other, MyMatrix):
            if self.shape != other.shape:
                return False
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if self.matrix[i][j] != other.matrix[i][j]:
                        return False
            return True
        else:
            raise ValueError("Can only compare with another MyMatrix instance")
    
    def inverse(self):
        n = len(self.matrix)
        AM = deepcopy(self.matrix)
        IM = [[int(i==j) for i in range(n)] for j in range(n)]
        indices = list(range(n))
        for fd in range(n):
            fdScaler = self._multiplicative_inverse(AM[fd][fd])
            for j in range(n): 
                AM[fd][j] = (AM[fd][j]*fdScaler) % self.p
                IM[fd][j] = (IM[fd][j]*fdScaler) % self.p
            for i in indices[0:fd] + indices[fd+1:]: 
                crScaler = AM[i][fd]
                for j in range(n): 
                    AM[i][j] = (AM[i][j] - crScaler * AM[fd][j]) % self.p
                    IM[i][j] = (IM[i][j] - crScaler * IM[fd][j]) % self.p
        return MyMatrix(IM, self.p)
    
    def inverse_float(self):
        n = len(self.matrix)
        AM = deepcopy(self.matrix)
        IM = [[int(i==j) for i in range(n)] for j in range(n)]
        indices = list(range(n))
        for fd in range(n):
            fdScaler = 1.0 / AM[fd][fd]
            for j in range(n): 
                AM[fd][j] *= fdScaler
                IM[fd][j] *= fdScaler
            for i in indices[0:fd] + indices[fd+1:]: 
                crScaler = AM[i][fd]
                for j in range(n): 
                    AM[i][j] -= crScaler * AM[fd][j]
                    IM[i][j] -= crScaler * IM[fd][j]
        return MyMatrix(IM)

    def mul(self, other):
        if isinstance(other, MyMatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("The number of columns of the first matrix must equal the number of rows of the second matrix")
            result = [[0 for _ in range(other.shape[1])] for _ in range(self.shape[0])]
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        result[i][j] += self.matrix[i][k] * other.matrix[k][j]
            return MyMatrix(result, self.p)
        elif isinstance(other, (int, float, complex)):
            result = [[0 for _ in range(self.shape[1])] for _ in range(self.shape[0])]
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i][j] = (self.matrix[i][j] * other)
            return MyMatrix(result, self.p)
        else:
            raise ValueError("Invalid type for multiplication")
        
    
    def T(self):
        transposed_matrix = [[self.matrix[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])]
        return MyMatrix(transposed_matrix, self.p)

def test():
    # Create two numpy arrays
    np_array1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np_array2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # Create two MyMatrix objects
    my_matrix1 = MyMatrix(np_array1)
    my_matrix2 = MyMatrix(np_array2)
    print(my_matrix1)
    print(my_matrix1.T())
    
    print()
    return
    my_matrix2 = MyMatrix(np_array2)
    # Create a list of lists
    list_of_lists = [[i * 7 + j for j in range(1, 8)] for i in range(1, 8)]

    # Create a MyMatrix object
    my_matrix = MyMatrix(list_of_lists)

    # Print the MyMatrix object
    print(my_matrix)


def generate_matrics(n=784, p=matrix_P):
    A_Matrix = []
    A_Matrix_ = []
    def generate_invertible_matrix_mod_p(n, p=matrix_P):
        # Generate a list of numbers that are coprime to p
        coprimes = [i for i in range(1, p) if sympy.gcd(i, p) == 1]
        # Generate a matrix with random entries coprime to p
        A = np.random.choice(coprimes, (n, n))
        return A
    numpy_matrix = generate_invertible_matrix_mod_p(n)
    test_matrix = MyMatrix(numpy_matrix)
    inverse_matrix = test_matrix.inverse()
    A_Matrix.append(test_matrix)
    A_Matrix_.append(inverse_matrix)
    with open(A_Matrix_path, "wb") as f:
        pickle.dump((A_Matrix, A_Matrix_), f)

def load_matrix():
    with open(A_Matrix_path, "rb") as f:
        return pickle.load(f)

def expand_matrix(nums):
    A_Matrix, A_Matrix_ = load_matrix()
    assert len(A_Matrix)>0 and len(A_Matrix_)>0 and len(A_Matrix)==len(A_Matrix_)
    if(len(A_Matrix) == nums):
        return A_Matrix, A_Matrix_
    tmp_A_Matrix = [A_Matrix[0] for i in range(nums)]
    tmp_A_Matrix_ = [A_Matrix_[0] for i in range(nums)]
    with open(A_Matrix_path, "wb") as f:
        pickle.dump((tmp_A_Matrix, tmp_A_Matrix_), f)
    return tmp_A_Matrix,tmp_A_Matrix_


def generate_matrics_float(n=784):
    A_Matrix = []
    A_Matrix_ = []
    
    def generate_invertible_integer_matrix(n, min_val=-100, max_val=100):
        while True:
            matrix = np.array([[np.random.randint(min_val, max_val) for _ in range(n)] for _ in range(n)])
            if np.linalg.matrix_rank(matrix) == n:
                return matrix
    
    numpy_matrix = generate_invertible_integer_matrix(n)
    numpy_inverse = np.linalg.inv(numpy_matrix)
    A_Matrix.append(numpy_matrix)
    A_Matrix_.append(numpy_inverse)
    with open(A_Matrix_float_path, "wb") as f:
        pickle.dump((A_Matrix, A_Matrix_), f)

def load_matrix_float():
    with open(A_Matrix_float_path, "rb") as f:
        return pickle.load(f)

def expand_matrix_float(nums):
    A_Matrix, A_Matrix_ = load_matrix_float()
    assert len(A_Matrix)>0 and len(A_Matrix_)>0 and len(A_Matrix)==len(A_Matrix_)
    if(len(A_Matrix) == nums):
        return A_Matrix, A_Matrix_
    tmp_A_Matrix = [A_Matrix[0] for i in range(nums)]
    tmp_A_Matrix_ = [A_Matrix_[0] for i in range(nums)]
    with open(A_Matrix_float_path, "wb") as f:
        pickle.dump((tmp_A_Matrix, tmp_A_Matrix_), f)
    return tmp_A_Matrix,tmp_A_Matrix_

def test2():
    # expand_matrix(60000)
    A_Matrix, A_Matrix_ = load_matrix()

    s = A_Matrix[0] * A_Matrix_[0]
    print(s)

def test3():
    A_Matrix, A_Matrix_ = load_matrix()
    Wh = MyMatrix(matrix = np.random.rand(784, 1))
    a = A_Matrix[0]
    a_ = A_Matrix_[0]
    Wh_ = a * (a_ * Wh)
    Wh1 = MyMatrix(matrix=np.random.randint(1,1000,size=(784, 1)))
    Wh1_ = a * (a_ * Wh1)
    return 


if __name__ == "__main__":
    n = 784
    # generate_matrics_float(n)
    # expand_matrix_float(60000)
    # generate_matrics(n)
    # expand_matrix(60000)
    
    test2()
    
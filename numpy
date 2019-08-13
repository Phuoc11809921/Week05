# Tạo mảng numpy (ndarray)
# tao ndarray
import numpy as np

# tao list
l = list(range(1, 4))

# tao ndarray
data = np.array(l)

print(l)
print(type(l))
print(data)
print(type(data))
print(data[0])
print(type(data[0]))
print(data[1])
print(data.shape)

# thay doi shape cua 1 mang
import numpy as np
arr1 = np.arange(12)
print(arr1)
print(arr1.shape)
print("---------------------")
arr2 = arr1.reshape((3,4))
print(arr2)
print(arr2.shape)

# Kieu du lieu
import numpy as np

# int32
arr1 = np.array([1,2])
print(arr1)
print(type(arr1))
print(arr1.dtype)

#float64
arr2 = np.array([1.0, 2.0])
print(arr2)
print(type(arr2))
print(arr2.dtype)

#int64
arr3 = np.array([1, 2], dtype=np.int64)
print(arr3)
print(type(arr3))
print(arr3.dtype)

# thay doi gia tri phan tu
import numpy as np

# tao list
l = list(range(1, 4))

# tao ndarray
data = np.array(l)
print(data)

data[0] = 8
print(data)
print("-----------------------")
# tao ndarray voi ham zeros
# shape: 2 dong, 3 cot
arr = np.zeros((2, 3))
print(arr)
print("-----------------------")
# numpy.ones(shape, dtype=none, order='C')
# shape: 2 dong, 3 cot
arr1 = np.ones((2, 3))
print(arr1)
print("-----------------------")
# numpy.full(shape, fill_value, dtype=none, order='C')
# shape: 2 dong, 3 cot
arr2 = np.full((2, 3), 9)
print(arr2)

# tao mot numpy array voi duong cheo la so 1
# So 0 duoc dien vao nhung cho trong con lai
import numpy as np

# numpy.eye(N, M=None, k=0, dtype=<class 'float'>, order='C')
# shape: 2 dong, 3 cot
arr = np.eye(5)
print(arr)
print("-----------------------")

# tao numpy array voi gia tri ngau nhien
# numpy.random.random(size=None)
# shape: 2 dong, 3 cot; voi cac phan tu co gia tri ngau nhien
arr1 = np.random.random((2, 3))
print(arr1)
print("-----------------------")

# Điều kiện cho mảng numpy
arr2 = np.arange(10)
print(arr2)

out = np.where(arr2 < 5, arr2, 10*arr2)
print(out)
print("-----------------------")

# Chuyển mảng về một chiều
arr3 = np.array([[1,2], [3,4]])
out = arr3.flatten()
print(arr3)
print(out)

# Lấy các phần tử từ mảng 2 chiều như sau

import numpy as np

# khoi tao numpy array co shape = (2, 3) co gia tri nhu sau:
# [[1 2 3]
#  [5 6 7]
a_arr = np.array([[1,2,3],[5,6,7]])
print('a_arr \n', a_arr)

# su dung slicing de tao mang b bang cach lay 2 hang dau tien
# va cot 1,2, Nhu vay b se co shape = (2, 2):
# [[2 3]
#  [6 7]]
b_arr = a_arr[:, 1:3]
print('\nb_arr \n', b_arr)
c_arr = a_arr[1:3, 1:3]
print('\nc_arr \n', c_arr)
print('\nbefore changing \n', a_arr[0, 1])
b_arr[0, 0] = 99
print('\nafter changing \n', a_arr[0, 1])
print('\na_arr \n', a_arr)

# Lấy 1 dong du lieu

import numpy as np

# khoi tao numpy array co shape = (3, 4) co gia tri nhu sau:
# [[1 2 3]
#  [5 6 7]
#  [9 10 11]]
arr = np.array([[1,2,3],[5,6,7],[9,10,11]])
print(arr)
print("-------------------------")
# 2 cach truy cap du lieu o hang giua cua mang 
# dung ket hop chi so va slice --> duoc array moi co so chieu thap hon
# neu chi dung slice ta se co array moi co cung so chieu voi array goc
#  Cach 1: so chieu giam
row_r1 = arr[1, : ]

# Cach 2: So chieu duoc giu nguyen
row_r2 = arr[1:2, : ]

print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print("-------------------------")
# Lấy 1 cot du lieu
col_r1 = arr[:, 1]
col_r2 = arr[:, 1:2]
print(col_r1, col_r1.shape)
print(col_r2, col_r2.shape)

import numpy as np

arr = np.array([[1, 2], [3, 4], [5, 6]])
print(arr)

# mot vi du cua truy xuat mang dung chi so (index)
# ket qua thu duoc la 1 mang co shape(3, )
print('\n', arr[[0, 1, 2], [0, 1, 0]])
# se thu duoc ket qua tuong duong nhu tren theo cach nay
print('\n', np.array([arr[0, 0], arr[1, 1], arr[2, 0]]))
# khi su dung index, ban duoc phep truy xuat toi 1 phan tu nhieu hon 1 lan
print('\n', arr[[0, 0], [1, 1]])
print("---------------------------")

# tim cac phan tu lon hon 2
# tra ve 1 mang boolean co so chieu nhu mang arr
# va gia tri tai moi phan tu la
# true neu phan tu cua a tai do > 2
# false cho truong hop nguoc lai
arr = np.array([[1, 2], [3, 4], [5, 6]])
print('\n',arr)
bool_idx = (arr > 2)
print('\nbool_idx\n',bool_idx)
# su dung boolean array indexing de xay dung mang 1 chieu
# bao gom cac phan tu tuong ung voi gia tri True cua bool_idx
# vi du o day in ra cac gia tri cua arr > 2, su dung array bool_idx da tao
out = arr[bool_idx]
print('\nmethod 1\n', out)
# mot cach ngan gon
print('\nmethod 1\n', arr[arr > 2])


# Phép toán trên mảng
import numpy as np

x = np.array([1,2,3,4], dtype=np.float64)
y = np.array([5,6,7,8], dtype=np.float64)

print('\ndata x \n', x)
print('\ndata y \n', y)

# Tong cua 2 mang, ca 2 cach cho cung 1 ket qua
print('\nmethod 1 \n', x + y)
print('\nmethod 2 \n', np.add(x, y))
print('-----------------------------------------')
# Hieu cua 2 mang, ca 2 cach cho cung 1 ket qua
print('\nmethod 1 \n', x - y)
print('\nmethod 2 \n', np.subtract(x, y))
print('-----------------------------------------')
# Nhan cua 2 mang, ca 2 cach cho cung 1 ket qua
print('\nmethod 1 \n', x * y)
print('\nmethod 2 \n', np.multiply(x, y))
print('-----------------------------------------')
# Chia cua 2 mang, ca 2 cach cho cung 1 ket qua
print('\nmethod 1 \n', x / y)
print('\nmethod 2 \n', np.divide(x, y))
print('-----------------------------------------')
# Can bac 2 tung phan tu trong x
print('\nsqrt \n', np.sqrt(x))

# Nhan giua 2 vector
v = np.array([1, 2])
w = np.array([2, 3])
# Tinh inner product giua 2 vector
print('\n method 1 \n', v.dot(w))
print('\n method 2 \n', np.dot(v,w))

# Nhân giữa vector và ma trận
X = np.array([[1,2],[4,3]])
v = np.array([1,2])

print('\n matrix X \n', X)
print('\n vector v \n', v)
# phep nhan giua ma tran va ventor
print('\n method 1: X.dot(v) \n', X.dot(v))
print('\n method 1: v.dot(X) \n', v.dot(X))
print('\n method 2: X.dot(v) \n', np.dot(X,v))
print('\n method 2: v.dot(X) \n', np.dot(v,X))

# Nhan giua 2 ma tran
X = np.array([[1,2],[4,3]])
Y = np.array([[2,3],[2,1]])

print(' matrix X \n', X)
print('\n matrix Y \n', Y)

# phep nhan giua hai ma tran
print('\n method 1 \n', X.dot(Y))
print('\n method 2 \n', np.dot(X, Y))

# Tính tổng cho một mảng numpy
x = np.array([[1,2],[3,4]])
# tong cac phan tu cua mang
print(np.sum(x))
# tinh tong theo tung cot
print(np.sum(x, axis=0))
# tinh tong theo tung hang
print(np.sum(x, axis=1))

# Chuyển vị cho một ma trận
x = np.array([[1,2],[3,4]])
print(X)
# chuyen vi
print(x.T)

# thực thi các phép toán trên các mảng có kích thước khác nhau.
X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
V = np.array([1,0,1])
Y = np.empty_like(X)

# Cach 1: Them vector v vao moi hang cua ma tran X bang mot vong lap
for i in range(4):
  Y[i, :] = X[i, :] + V
  
print('\n Matrix X \n', X)
print('\n Matrix V \n', V)
print('\n Matrix Y \n', Y)

# Khi ma trận X lớn, việc sử dụng vòng lặp này sẽ rất chậm
# xếp chồng nhiều bản sao của v theo chiều dọc, sau đó thực hiện phép tính tổng với X
X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
V = np.array([1,0,1])

# xep chong 4 ban sao cua v len nhau
V_t = np.tile(V, (4, 1))
# thuc hien phep cong
Y = X + V_t

print('\n Matrix X \n', X)
print('\n Matrix V \n', V)
print('\n Matrix V_t \n', V_t)
print('\n Matrix Y \n', Y)
print('---------------------------')
# Numpy broadcasting cho phép chúng ta thực thi tính toán này mà không cần phải làm thêm các bước thêm nào.
X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
V = np.array([1,0,1])
Y = X + V
print(Y)

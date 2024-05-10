#!/usr/bin/env python
# coding=utf-8
import numpy as np

def test_arr():
    arr = [1,2,3,4,5,6]
    print(type(arr))
    print(arr)

    arr2 = np.array(arr)
    print(type(arr2))
    print(arr2)

def test_tuple():
    tuple = (1,2,3,4,5,6)
    print(type(tuple))
    print(tuple)

    tuple_np = np.array(tuple)
    print(type(tuple_np))
    print(tuple_np)

    arr = [1,2,3,4,5,6]
    print(type(arr))
    print(arr)

    arr2 = np.array(arr)
    print(type(arr2))
    print(arr2)

def test_equle():
    array = np.array([1,2,3])
    tuple = np.array((1,2,3))

    print(array == tuple)

def test_2_arr():
    arr1 = [[1,2],[3,4]]
    print(arr1)

    arr2 = np.array(arr1)
    print(arr2)

def test_arange():
    arr1 = np.arange(100)
    print(arr1)

    arr2 = np.array([np.arange(5),np.arange(5)])
    print(arr2)

def test_shape():
    arr1 = np.arange(5)
    print(arr1)

    #reshape不改变原数据
    arr2 = arr1.reshape(5,1)
    print(arr2)

    arr3 = arr1.reshape(1,5)
    print(arr3)

    # resize改变原数据
    print(arr1)
    arr4 = arr1.resize(5,1)
    print(arr1)

    arr5 = arr1.resize(1,5)
    print(arr1)

def test_type():
    arr1 = [1,2,3,4]
    print(arr1)

    arr2 = np.int8(arr1)
    print(arr2)

    arr3 = np.int16(arr1)
    print(arr3)
    
    arr4 = np.float32(arr1)
    print(arr4)

    arr5 = np.float64(arr1)
    print(arr5)

    new_arr1 = np.arange(100,dtype=float)
    print(new_arr1)

    new_arr2 = np.arange(100,dtype='float32')
    print(new_arr2)
    
    new_arr3 = np.arange(100,dtype=int)
    print(new_arr3)

    new_arr4 = np.arange(100,dtype='int16')
    print(new_arr4)

    #复数
    new_arr5 = np.arange(100,dtype='D')
    print(new_arr5)
    
def test_param():
    arr1 = np.array([1,2,3,4,5,6,7,8,9])
    arr2 = np.array([[1,2,3],[4,5,6]])
    arr3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    arr4 = np.array([[[1],[2],[3]],[[4],[5],[6]]])
    print(arr1)
    print(arr2)
    print(arr3)

    #ndim:数组维度
    #1维数字
    print(arr1.ndim)
    #2维数字
    print(arr2.ndim)
    #2维数字
    print(arr3.ndim)
    #3维数字
    print(arr4.ndim)

    #shape:列数和行数,类型tuple(列数,行数)
    print(arr1.shape)
    print(arr2.shape)
    print(arr3.shape)

    #size总数量，行数x列数
    print(arr1.size)
    print(arr2.size)
    print(arr3.size)

    #itemsize:单个元素字节数
    #8,从python array转过来的，默认是8字节
    print(arr1.itemsize)

    size_arr1 = np.arange(10,dtype='int8')
    #1,int8位=1字节
    print(size_arr1.itemsize)

    size_arr2 = np.arange(10,dtype='int16')
    #2,int16位=2字节
    print(size_arr2.itemsize)

    size_arr3 = np.arange(10,dtype='int')
    #8,系统相关，64位系统，整型，即8字节
    print(size_arr3.itemsize)

    size_arr4 = np.arange(10,dtype='float')
    #8,系统相关，64位系统，浮点型，即8字节
    print(size_arr4.itemsize)

    size_arr5 = np.arange(10,dtype='float32')
    #4,32位即4字节
    print(size_arr5.itemsize)

    size_arr6 = np.arange(10,dtype='float64')
    #8,64位即8字节
    print(size_arr6.itemsize)


    #nbytes：总字节数量 = itemsize x size
    print(size_arr1.nbytes)
    print(size_arr2.nbytes)
    print(size_arr3.nbytes)
    print(size_arr4.nbytes)
    print(size_arr5.nbytes)
    print(size_arr6.nbytes)

def test_index():
    arr1 = np.arange(100)
    #正数从前数，负数从后数，-1 = 100,-10 = 90
    arr2 = arr1[:10]
    arr3 = arr1[10:]
    arr4 = arr1[:]
    arr5 = arr1[-1:]
    arr6 = arr1[-10:]
    arr7 = arr1[-10:-1]
    arr8 = arr1[:-1]
    arr8 = arr1[:-10]
    print(arr2)
    print(arr3)
    print(arr4)
    print(arr5)
    print(arr6)
    print(arr7)
    print(arr8)

    arr_2_1 = np.array([np.arange(10),np.arange(10)])
    arr_2_2 = arr_2_1[[0]]
    print(arr_2_2)

def test_flatten():
    arr1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #转为1维数组，返回拷贝，修改不影响
    arr2 = arr1.flatten()
    arr2[0]=np.array(100)
    print(type(arr2[0]))
    #赋值是100会自动变称numpy.int64类型，与上面写法结果一样
    #arr2[0]=100
    #print(type(arr2[0]))
    print(arr1)
    print(arr2)

    arr1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    #转为1维数组，返回引用，改变原数组
    arr2 = arr1.ravel()
    arr2[0]=np.array(100)
    print(arr1)
    print(arr2)

def test_shape_tuple():
    arr1 = np.arange(100)

    #必须是行x列=总，2x50=100，否则报错
    arr1.shape = (2,50)
    print(arr1)
    arr1.shape = (25,4)
    print(arr1)

def test_transpose():
    #转制，必须2为以上数组
    arr1 = np.array([np.arange(4),np.arange(4)])
    print(arr1)
    arr2 = arr1.transpose()
    #不影响原数组
    print(arr1)
    print(arr2)

    #也可以直接使用T属性
    print(arr1.T)

def test_add():
    arr1 = np.array([np.arange(4),np.arange(4)])
    arr2 = np.arange(16)
    print(arr1)
    print(arr2)

    print(arr1 * 2)
    print(arr1 + 2)
    print(arr1 / 2)

    print(arr2 * 2)
    print(arr2 + 2)
    print(arr2 / 2)

    #不影响原数组，都是拷贝
    print(arr1)
    print(arr2)

def test_stack():
    #垂直相连
    arr1 = np.arange(4)
    arr2 = np.arange(4)
    arr3 = np.vstack((arr1,arr2))
    #等同于vstack
    arr4 = np.row_stack((arr1,arr2))
    print(arr1)
    print(arr2)
    print(arr3)
    print(arr4)

    #水平相连
    arr1 = np.arange(4)
    arr2 = np.arange(4)
    arr3 = np.hstack((arr1,arr2))
    #等同于hstack
    arr4 = np.column_stack((arr1,arr2))
    print(arr1)
    print(arr2)
    print(arr3)
    print(arr4)

    #concatenate，通过axis控制
    arr1 = np.arange(4)
    arr2 = np.arange(4)
    arr3 = np.array([arr1,arr2]) * 2
    arr4 = np.array([arr1,arr2])
    #axis=0水平相加
    arr5 = np.concatenate((arr3,arr4),axis=1)
    #axis=1垂直相加
    arr6 = np.concatenate((arr3,arr4),axis=0)
    print(arr3)
    print(arr4)
    print(arr5)
    print(arr6)

    #dstack深度相加,两个纬数组变成3纬数组
    arr1 = np.arange(4)
    arr2 = np.arange(4)
    arr3 = np.array([arr1,arr2]) * 2
    arr4 = np.array([arr1,arr2])
    arr5 = np.dstack((arr3,arr4))
    print(arr3)
    print(arr3.shape)
    print(arr4)
    print(arr4.shape)
    print(arr5)
    print(arr5.shape)

def test_split():
    arr1 = np.arange(24)
    arr1.shape = (4,6)
    print(arr1)

    #拆分结果都是包含结果的数组

    #水平拆分，hsplit = split(axis=1)
    arr2 = np.hsplit(arr1,2)
    print(arr2)
    arr3 = np.split(arr1,2,axis=1)
    print(arr3)

    #垂直拆分，vsplit = split(axis=0)
    arr2 = np.vsplit(arr1,2)
    print(arr2)
    arr3 = np.split(arr1,2,axis=0)
    print(arr3)

def test_tolist():
    arr1 = np.arange(10)
    arr2 = arr1.reshape((2,5))

    arr3 = arr1.tolist()
    arr4 = arr2.tolist()
    print(arr1)
    print(arr3)
    print(arr2) 
    print(arr4)

def test_astype():
    arr1 = np.arange(10)
    arr2 = arr1.astype(float)
    arr3 = arr1.astype('float')
    arr4 = arr1.astype('byte')
    
    print(arr1)
    print(arr2)
    print(arr3)
    print(arr4)

def test_func():
    arr1 = np.arange(10)
    arr2 = np.arange(10) * 2

    #sum
    arr3 = np.sum(arr1)
    arr4 = np.sum(arr2)
    arr5 = np.sum((arr1,arr2))
    print(arr3)
    print(arr4)
    print(arr5)

    #mean 均值
    arr6 = np.mean(arr1)
    print(arr6)

    #max
    arr7 = np.max(arr1)
    print(arr7)

    #min
    arr8 = np.min(arr1)
    print(arr8)

    #ptp max-min
    arr9 = np.ptp(arr1)
    print(arr9)

    #std 标准变差,平均值一样时看谁更佳离散
    arr0 = np.array([2,0])
    arr10 = np.std(arr0)
    print(arr10)

    #var 方差
    arr0 = np.array([2,0])
    arr11 = np.var(arr0)
    print(arr11)

    #cumsum累加
    arr12 = np.cumsum(arr1)
    #水平方向累加，与上等同
    arr13 = np.cumsum(arr1,axis=0)
    print(arr12)
    print(arr13)

    #cumprod累乘
    arr14 = np.cumprod(arr1+1)
    print(arr14)

def test_zeros():
    arr1 = np.zeros(10)
    print(arr1)

def test_random():
    arr1 = np.ones([2,4])
    print(arr1)

if __name__ == '__main__':
    test_random()






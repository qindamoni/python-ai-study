#!/usr/bin/env python
# coding=utf-8
import torch
import numpy

def test_create():
    #empty值由气筒决定
    a1 = torch.empty(5)
    print(type(a1))
    print(a1)
    
    a2 = torch.empty(1,5)
    print(a2)
    print(a2[0])

    #随机生成
    a3 = torch.empty(5,1,dtype=torch.int)
    print(a3)
    print(a3[0])

    a4 = torch.rand(5)
    print(a4)

    a5 = torch.rand(5,3)
    print(a5)

    #默认0
    a6 = torch.zeros(5,dtype=torch.double)
    print(a6)

    a7 = torch.zeros(5,3)
    print(a7)

    #use python array
    arr1 = [1,2,3,4,5]
    a8 = torch.tensor(arr1,dtype=torch.double)
    print(a8)

    a9 = torch.tensor(arr1)
    print(a9)

    #use old one
    a10 = torch.zeros(3,4)
    a11 = a10.new_ones(4,10,dtype=torch.int)
    print(a10)
    print(a11)

    a12 = torch.randn_like(a10,dtype=torch.float)
    print(a12)

    #ones,device='cuda'
    a13 = torch.ones_like(a12,device='cpu')
    print(a13)

def test_size():
    a1 = torch.ones(4)
    print(a1)
    #tuple
    print(a1.size())

    a2 = torch.zeros(10,5)
    print(a2.size())

def test_add():
    a1 = torch.ones(4,3)
    a2 = torch.ones(4,3)
    a3 = a1+a2
    #等同于+
    a4 = torch.add(a1,a2)
    print(a3)
    print(a4)

    #引用添加
    a5 = torch.empty(4,3)
    torch.add(a1,a2,out=a5)
    print(a5)

    #引用添加，会修改a1值,add_()，copy_()等有下划线的函数都会影响a1的值
    a1.add_(a2)
    print('--------------')
    print(a1)

    #其他运算
    print(a1 * 3)
    print(a1 + 3)
    print(a1 - 3)
    print(a1 / 3)
    print(a1 % 3)

def test_split():
    a1 = torch.tensor([1,2,3,4,5])
    print(a1)

    a2 = a1[0:2]
    print(a2)

    a2 = a1[2:-1]
    print(a2)

def test_shape():
    a1 = torch.tensor([1,2,3,4,5,6,7,8])
    print(a1)
    print(a1.size())

    #行x列必须等于size，2x4=8
    a2 = a1.view(2,4)
    print(a2)
    print(a2.size())

    #-1表示自行推算列数
    a3 = a1.view(4,-1)
    print(a3)
    print(a3.size())

def test_item():
    a1 = torch.tensor([1,2,3,4,5])
    print(a1)

    a2 = a1[0]
    print(a2)
    print(type(a2))

    a3 = a2.item()
    print(type(a3))

def test_numpy():
    a1 = torch.tensor([1,2,3,4,5])
    print(a1)
    print(type(a1))

    #tensor to numpy
    a2 = a1.numpy()
    print(a2)
    print(type(a2))

    a1.add_(1)
    print(a1)
    #引用，会被修改
    print(a2)

    #numpy to tensor
    n1 = numpy.ones(5)
    print(n1)

    a1 = torch.from_numpy(n1)
    print(a1)

def test_device():
    if torch.cuda.is_available():
        a1 = torch.rand(4,3,device='cuda')

    a10 = torch.rand(4,3,device='cpu')
    print(a10)

    device=torch.device('cpu')
    a11 = torch.rand(4,3,device=device)
    print(a11)

    a12 = a10.to(device)
    print(a12)

if __name__ == '__main__':
    test_device()

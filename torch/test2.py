#!/usr/bin/env python
# coding=utf-8
import torch

def test_data():
    t1 = torch.tensor([1,2,3,4,5,6,7,8,9,10])
    t2 = t1.view(-1,2)

    print(t2.size())
    print(t2)

    t3 = t2[:1]
    print(t3)
    print(t3.size())

    t2 = torch.cat([t1,t1])
    print(t2)

    t3 = t2.unsqueeze(-1)
    print(t3)


if __name__ == '__main__':
    test_data()

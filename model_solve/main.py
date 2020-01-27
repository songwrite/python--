#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/1/26 下午8:11
@email: 2444594190@qq.com
@author: 宋腾
"""
from model_solve.data_struct import Solver
import numpy as np

#bound_condition,init_condition,k,population_density,w
M1 = 8
M2 = 7
N = 10

if __name__ == '__main__':
    #边界条件
    bc = (np.zeros((N,M1)),
          np.zeros((N,M1)),
          np.zeros((N,M2)),
          np.zeros((N,M2))
    )
    #初始条件
    ic = np.zeros((M1,M2))
    ic[2,3] = 3
    #人口流动系数
    k = 0.001+0.9*np.random.random((N,M1,M2))
    #人口密度
    pop_density = 10000+1000*(np.random.random((N,M1,M2))-0.5)
    #感染率
    w = 0.0001*np.ones((N,M1,M2))

    #调用模型求解
    solver = Solver(bc,ic,k,pop_density,w)
    R = solver.solve_model(1,2)
    R = np.floor(R)
    R = np.array(R,dtype=int)
    print(R)
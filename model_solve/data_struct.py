#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2020/1/26下午6:55
@email: 2444594190@qq.com
@author: 宋腾
"""


from numpy import array,zeros_like


def diff(f,i,j,n,h,dim = 'x'):
    '''求(i,j,n)处的对dim的偏导数'''
    if dim == 'x':
        d = (f[n,i+1,j]-f[n,i-1,j])/(2*h)
    elif dim == 'y':
        d = (f[n,i,j+1]-f[n,i,j-1])/(2*h)
    else:
        raise  ValueError('dim只能为x或y')
    return d


class Solver:

    def __init__(self,bound_condition: array,init_condition: array,k: array,population_density: array,w: array):
        '''
        参数：
        bound_condition=(left,right,up,down)边界条件,每个方向的边界条件均为二维numpy.array,每行一个时刻的边界条件(初始时刻无须边界条件,共N-1行)
        init_condition 初始条件,为一个二维numpy.array,保存初始时刻的患病人数
        k,population_density,w  分别对应人口流动系数、人口密度、感染率,均为三维numpy.array。
        '''
        self.__bound_condition = bound_condition
        self.__init_condition = init_condition
        self.__k = k
        self.__population_density = population_density
        self.__w = w


    def solve_model(self,tau,h):
        '''
        参数：
        tau 时间步长
        h 空间步长
        '''
        R = zeros_like(self.__population_density)
        N,M1,M2 = R.shape
        R[0,:,:] = self.__init_condition #设置初始条件
        for n in range(1,N):
            #设置边界条件
            R[n,:,0] = self.__bound_condition[0][n,:]
            R[n,:,-1] = self.__bound_condition[1][n,:]
            R[n,0,:] = self.__bound_condition[2][n,:]
            R[n,-1,:] = self.__bound_condition[3][n,:]
            #迭代计算
            for i in range(1,M1-1):
                for j in range(1,M2-1):
                    R[n,i,j] = R[n-1,i,j]+self.__k[n-1,i,j]*tau/h**2* \
                               (R[n-1,i+1,j]+R[n-1,i-1,j]-2*R[n-1,i,j]+\
                                R[n-1,i,j+1]+R[n-1,i,j-1]-2*R[n-1,i,j])+ \
                               tau*(diff(self.__k,i,j,n-1,h,'x')+diff(self.__k,i,j,n-1,h,'y'))/h* \
                               (R[n-1,i+1,j]+R[n-1,i,j+1]-2*R[n-1,i,j])+\
                        self.__w[n-1,i,j]*self.__population_density[n-1,i,j]*R[n-1,i,j]*tau
        return R


if __name__ == '__main__':
    pass
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:22:00 2023

@author: mimor
"""
import numpy as np
time = 1000
dt = 0.001
dx = 10
D = 0.4
d =D/(dx**2)
A_n = 11
a = np.zeros(A_n)
a[5] = 1000
t_total =0
def propensity(a,dx,D,dt):
    d =D/(dx**2)
    a0 = sum(a[:-1]*d)+sum(a[1:]*d)
    return(a0)

while t_total <=time:
    tau  =(1/propensity(a=a, dx=10, D=0.4, dt=0.001))*np.log(1/np.random.uniform(0,1))
    
    r= np.random.uniform(0,1)
    #move to the right
    diff_right = np.cumsum(a[:-1]*d)/(sum(a[:-1]*d+sum(a[1:]*d)))
    pr = np.where(r <= diff_right)[0]
    if any(pr):
        a[pr[0]]-=1
        a[pr[0]+1]+=1
        
    #move to the left
    else:
        diff_left = np.cumsum(a[1:]*d)/(sum(a[:-1]*d+sum(a[1:]*d)))+diff_right[-1]
        pl  =np.where(r <=diff_left)[0]
        if any(pl):
            a[pl[0]+1]-=1
            a[pl[0]]+=1
    t_total += tau
    tau = 0

print(a)




































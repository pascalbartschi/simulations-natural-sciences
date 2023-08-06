#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:59:35 2023

@author: lukas
"""

import numpy as np


triangle_initial = np.asarray([[0,0],
                               [2.0e-6,0.0e-6],
                               [1.0e-6,1.7e-6]])

triangle_afterwards = np.asarray([[0,0],
                                  [2.3e-6,0.2e-6],
                                  [1.8e-6,1.8e-6]])


E = 15 #Nm
v = 0.7


def calulcate_tensor(triangle_initial, triangle_afterwards):
    
    change_first_point = triangle_afterwards[0] - triangle_initial[0]
    triangle_afterwards = triangle_afterwards - change_first_point
    
    
    dx = np.zeros(2)
    dx[0] = triangle_afterwards[1][0] - triangle_initial[1][0]
    dx[1] = triangle_afterwards[2][0] - triangle_initial[2][0]
    
    x = np.zeros((2,2))
    x[0] = triangle_initial[1] - triangle_initial[0]
    x[1] = triangle_initial[2] - triangle_initial[0]
    
    dy = np.zeros(2)
    dy[0] = triangle_afterwards[1][1] - triangle_initial[1][1]
    dy[1] = triangle_afterwards[2][1] - triangle_initial[2][1]
    
    sol1 = np.linalg.solve(x,dx)
    sol2 = np.linalg.solve(x,dy)
    
    #Gleichungssystem 
    # 1.dux = Exx*x + Exxy*y = Exx*x + dux/dy *y
    # 2.dux = Eyy*y + Eyyx*y = Eyy*x + duy/dx *x
    
    Exx = sol1[0]
    Eyy = sol2[1]
    Exy = 0.5*(sol1[1]+sol2[0])
    
    # Exx = round(sol1[0],2)
    # Eyy = round(sol2[1],2)
    # Exy = round(0.5*(sol1[1]+sol2[0]),2)
    
    tensor = np.array([[Exx,Exy],
                   [Exy,Eyy]])
    
    return tensor


def calculate_stress(strain_sensor,E,v):
    matrix_pre = np.asarray([[1,v,0],
                         [v,1,0],
                         [0,0,1-v]])
    preterm = E/((1+v)*(1-v))
    strain_tensor_short = np.asarray([strain_tensor[0][0],
                            strain_tensor[1][1],
                            strain_tensor[0][1]])
    #stress_tensor_short = preterm*np.dot(matrix_pre,strain_tensor_short)
    stress_tensor_short = preterm*np.matmul(matrix_pre,strain_tensor_short)
    stress_tensor = np.asarray([[stress_tensor_short[0],stress_tensor_short[2]],
                                [stress_tensor_short[2],stress_tensor_short[1]]])
    return(stress_tensor)



strain_tensor = calulcate_tensor(triangle_initial, triangle_afterwards)
stress_tensor = calculate_stress(strain_tensor,E,v)

print(strain_tensor)
print(stress_tensor)


#%%


def energy(triangle_initial, triangle_afterwards):
    strain_tensor = calulcate_tensor(triangle_initial, triangle_afterwards)
    
    print(f"ST {strain_tensor}")

    
    Exx = strain_tensor[0][0]
    Eyy = strain_tensor[1][1]
    Exy = strain_tensor[1][0]
    
    lame = v*E/((1+v)*(1-v))
    mu = E/(2*(1+v))
    # print(f"exx {Exx} \n eyy {Eyy} \n exy {Exy}")
    
    U = 0.5*lame*(Exx+Eyy)**2+mu*(Exx**2+Eyy**2+2*Exy**2)
    print(f"U {U}")
    
    
    x1 = triangle_initial[0][0]
    x2 = triangle_initial[1][0]
    x3 = triangle_initial[2][0]
    
    y1 = triangle_initial[0][1]
    y2 = triangle_initial[1][1]
    y3 = triangle_initial[2][1]
    
    A= 0.5*abs(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)#*10**-12 #Units

    Energy = U*A
    # print(f"Energy")
    
    return Energy


energy_ori = energy(triangle_initial, triangle_afterwards)

dx = 1e-12

triangle_x = np.copy(triangle_afterwards)
triangle_x[0][0] +=  dx

energy_x = energy(triangle_initial, triangle_x)


triangle_y = np.copy(triangle_afterwards)
triangle_y[0][1] +=  dx

energy_y = energy(triangle_initial, triangle_y)

change = np.array([energy_x-energy_ori,energy_y-energy_ori]) / dx

Force = change*-1

print(Force)



# Area 1.7*e-12




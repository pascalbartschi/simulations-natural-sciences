# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 13:00:30 2023

@author: paesc
"""
import numpy as np

# EXERCISE
# A


def calc_strain_tensor(old_coord, new_coord):
    
    # change_first_point = new_coord[0] - old_coord[0]
    # triangle_afterwards = new_coord - change_first_point
    

    # for function to work one point is fixed, as this doesnt change strain, but wrong correction
    
    # corr = np.array([new_coord[0] - old_coord[0],
    #                   new_coord[0] - old_coord[0], 
    #                   new_coord[0] - old_coord[0]])
    # print('corr', corr)
    # # Thus, correction is applied if reference point is shifted, point zero in my case
    # # new_coord_copy = new_coord.copy()
    # new_coord -= corr # with correction other forces are trash
    # print('new_coor', new_coord)
    # corr = new_coord[0] - old_coord[0]

    # assumption first point is always fixed
    dx = old_coord[1:, 0] # - np.array([old_coord[0, 0], old_coord[0, 0]])
    dy = old_coord[1:, 1] # - np.array([old_coord[0, 1], old_coord[0, 1]])
    
    dux = new_coord[1:, 0] - old_coord[1:, 0]
    duy = new_coord[1:, 1] - old_coord[1:, 1] 
    
    e_xx = np.diff(dy * np.roll(dux,1)) / np.diff(np.roll(dx, 1) * dy)[0]
    e_yy = np.diff(dx * np.roll(duy,1)) / np.diff(np.roll(dy, 1) * dx)[0]
    
    e_xxy =  np.diff(dx * np.roll(dux,1)) / np.diff(np.roll(dy, 1) * dx)[0]
    e_yyx =  np.diff(dy * np.roll(duy,1)) / np.diff(np.roll(dx, 1) * dy)[0]
    
    e_xy = 0.5 * (e_xxy + e_yyx)
    
    return np.array([[e_xx, e_xy], [e_xy, e_yy]]) # fuse strain tensor together


# B
def calc_stress_tensor(E, v, strain_tensor):
    
   
    strain_vector = np.array([strain_tensor[0, 0], strain_tensor[1, 1], strain_tensor[0, 1]])
    factor = E / ((1 + v) * (1 - v))
    conversion_M = np.array([[1, v, 0], [v, 1, 0], [0, 0, 1-v]])
    
    sigma_x, sigma_y, sigma_xy = factor * np.matmul(conversion_M, strain_vector)

    return np.array([[sigma_x[0], sigma_xy[0]], [sigma_xy[0], sigma_y[0]]])



def shearing_modulus(E, v):
    
    return E / (2 * (1 + v))

def lames_param(E, v):
    return v * E / ((1 + v) * (1 - v))

def calc_area(coords):
    
    return -0.5 * (coords[:, 0] * np.roll(coords[:, 1], 1) - np.roll(coords[:, 0], 1) * coords[:, 1]).sum()


def calc_energy(old_coord, new_coord, v, E):
    
    mu = shearing_modulus(E, v)
    lam = lames_param(E, v)
    
    
    strain_tensor = calc_strain_tensor(old_coord = old_coord,
                                        new_coord = new_coord)
    
    # strain_tensor = calulcate_tensor(old_coord, new_coord)
    # print(f"ST{strain_tensor}")
    
    # stress_tensor = calc_stress_tensor(E = E,
    #                            v = v,
    #                            strain_tensor = strain_tensor)
    
    e_xx, e_yy, e_xy  = strain_tensor[0, 0][0],  strain_tensor[1, 1][0],  strain_tensor[1, 0][0]
    # print(f"exx {e_xx} \n eyy {e_yy} \n exy {e_xy}")
    U = 0.5 * lam * (e_xx + e_yy) ** 2 + mu * (e_xx ** 2 + e_yy ** 2 + 2 * e_xy ** 2) 
    # print(f"U {U}")
    
    return U * calc_area(old_coord)







# C: forces on the triangles
# forces = np.array([[4.81, 3.35], [-2.13, -3.09], [-2.69, -0.26]])


# s_xx, s_yy, s_xy  = strain_tensor[0, 0][0],  stress_tensor[1, 1][0],  stress_tensor[1, 0][0]

# 0.5 * s_xx  ** e_xx + 0.5 * s_yy * e_yy + 0.5 * s_xy * e_xy
# U = 0.5 * lam * (e_xx + e_yy) ** 2 + mu * (e_xx ** 2 + e_yy ** 2 + 2 * e_xy ** 2) 

# new and old coordinates counter clockwise
old_coord = np.array([[0., 0.], [2., 0.], [1., 1.7]]) * 1e-6
new_coord = np.array([[0., 0.1], [2.3, 0.2], [1.8, 1.8]]) * 1e-6
# parameters
E = 15   # Nm^-1
v = 0.7 


E_ori = calc_energy(old_coord, new_coord, v, E)# U * area(old_coord)

delta = 1e-12
# len(old_coord)
results = np.zeros_like(old_coord)
# len(old_coord)

for i in range(len(old_coord)):
    
    if i == 0: # instead of
        # shift point for delta to obtain gradient
        new_x = new_coord.copy()
        new_x[1, 0] -= delta
        new_x[2, 0] -= delta
        new_y = new_coord.copy()
        new_y[1, 1] -= delta
        new_y[2, 1] -= delta
        
    else:
        # shift point for delta to obtain gradient
        new_x = new_coord.copy()
        new_x[i, 0] += delta
        new_y = new_coord.copy()
        new_y[i, 1] += delta
    
    
    Enew_x = calc_energy(old_coord, new_x, v, E)
    force_x = (Enew_x - E_ori) / delta

    Enew_y = calc_energy(old_coord, new_y, v, E)
    force_y = (Enew_y - E_ori) / delta

     # force is minus energy gradient
    results[i] = [-force_x, -force_y]
    
    
print(results)






















# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 16:03:31 2015

@author: tinria
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

#set parameters
s=0.375 #strain factor
k=1     #spring constant
kA=10   #area elasticity constant
zeta=1  #viscous coefficient
m=1     #node mass

ttot=100    #total simulated time
dt=0.1      #delta t

#defining initial vertex positions
#VP=vertex positions; VP[:,0]: x-positions; VP[:,1]: y-positions

VP=np.zeros([42,2])
for i in range(18):
    VP[i,0]=3.6-i*1.15/17
    VP[i,1]=29.75-i*29/17
VP[18,0]=0.86
VP[18,1]=0
VP[19,0]=-0.86
VP[19,1]=0
for i in range(20,38):
    VP[i,0]=-2.45-(i+1-21)*1.15/17
    VP[i,1]=0.75+(i+1-21)*29/17
VP[38,0]=-2.6
VP[38,1]=31.2
VP[39,0]=-0.86
VP[39,1]=31.4
VP[40,0]=0.86
VP[40,1]=31.4
VP[41,0]=2.6
VP[41,1]=31.2



# calculating edge (spring) lengths, given the positions of the vertices    
def edge_lengths(VP):
    l=np.zeros([42])
    l[0:41]=((VP[1:42,0]-VP[0:41,0])**2+(VP[1:42,1]-VP[0:41,1])**2)**0.5
    l[41]=((VP[0,0]-VP[41,0])**2+(VP[0,1]-VP[41,1])**2)**0.5
    return l

# calculating the spring forces on the vertices (Fx: x-direction, Fy: y-direction), 
# given the vertex positions, the rest lengths, and the spring constant
def spring_forces(VP,l0,k):
    """
    Hooke's law, the idea is to use the fraction of dx/l = Fx / F to get the seperate forces in Fx and Fy -> Trigonmetry
    :param VP: arrray of verticepos
    :param l0: resting length
    :param k: spring konstant
    :return: Fx, Fy
    """
    # first clock wise
    l = edge_lengths(VP)
    Ftot = -k * (l - l0) / l0   # total force
    # force in the + direction
    dx1 = VP[:, 0] - np.roll(VP[:, 0], -1)
    dy1 = VP[:, 1] - np.roll(VP[:, 1], -1)
    Fx1 = Ftot * (dx1/l)
    Fy1 = Ftot * (dy1/l)
    # print(f"Fx1: {Fx1}")
    
    
    # counter clockwise
    # force in the - direction
    # dx2 = VP[:, 0]np.roll(VP[:, 0], 1)
    # dy2 = VP[:, 1] - np.roll(VP[:, 1], 1) 
    # Fx2 = Ftot * (dx2/l)
    # Fy2 = Ftot * (dy2/l)
    Fx2 = - np.roll(Fx1, 1)
    Fy2 = - np.roll(Fy1, 1)
    # print(f"Fx2: {Fx2}")
    
    Fx = Fx2 + Fx1
    Fy = Fy2 + Fy1
    print(f"Fx: {Fx}\n Fy: {Fy}")

    return Fx,Fy

# calculating the area of a polygon, given the positions of its vertices
def polygon_area(VP):
    area=-0.5*sum(VP[0:41,0]*VP[1:42,1]-VP[1:42,0]*VP[0:41,1])
    area-=0.5*(VP[-1,0]*VP[0,1]-VP[0,0]*VP[-1,1])
    return area
  
# calculating the forces (FAx and FAy) on vertices arising from area elasticity,
# given the vertex positions, the area elasticity constant, and the target area  
def area_forces(VP,kA,A0):
    A = polygon_area(VP)
    A_k = (kA / 2) * ((A-A0) / A0**2)
    FAx = A_k * (np.roll(VP[:, 1], -1) - np.roll(VP[:, 1], 1)) # for force in x take the y positions, as eq is derived in x
    FAy = A_k * (np.roll(VP[:, 0], 1) - np.roll(VP[:, 0], -1))
    # print(f"FAx: {FAx}\n FAy: {FAy}")

    return FAx,FAy

#velocity in y-direction
#calculate values to start simulation
A0=polygon_area(VP)     #target area
L0=edge_lengths(VP)     #L0: edge lengths
l0=(1-s)*L0             #l0: resting lengths
vx=np.zeros(len(L0))    #velocity in x-direction
vy=np.zeros(len(L0)) 
fig = plt.figure()
panel = fig.add_subplot(1,1,1)

#simulation part  int(ttot/dt)
for i in range(int(ttot/dt)):
    Fx,Fy=spring_forces(VP,l0,k)
    FAx,FAy=area_forces(VP,kA,A0)
    Fxtot=Fx+FAx
    Fytot=Fy+FAy
    ax=(Fxtot-zeta*vx)/m #ax: acceleration in x-direction
    ay=(Fytot-zeta*vy)/m #ay: acceleration in y-direction
    VP[:,0]+=vx*dt+0.5*ax*dt**2
    VP[:,1]+=vy*dt+0.5*ay*dt**2
    vx+=ax*dt
    vy+=ay*dt
    
    if i%10 == 0:
        panel.clear()
        panel.set_aspect('equal')
        panel.set_ylim(-3,35)
        panel.set_xlim(-15,15)
        panel.plot(VP[:,0],VP[:,1],'blue')
        panel.plot(VP[[0,-1],0],VP[[0,-1],1],'blue')
        plt.pause(0.01)
    if i%100 == 0:
        print ('step:',i,'from',int(ttot/dt))

plt.show()
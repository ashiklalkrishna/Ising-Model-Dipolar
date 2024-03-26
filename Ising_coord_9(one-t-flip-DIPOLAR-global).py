import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.spatial import cKDTree
import time
import sys
import pandas as pd
s_time=time.time()

def ising_lattice (N, a):
    coord = np.zeros((N**2, 2), dtype=float)
    angle = np.zeros(N**2, dtype=float)
    y=0
    i=0
    nrows = N
    ncols = N

    for row in range(0, nrows):
        x=0
        for col in range (0, ncols):
            coord[i] = [x, y]
            angle[i] = np.pi/2
            x += a

            i += 1
        y += a
    
    return coord, angle

def initial(N, frac):
    rand_grid=np.random.random(N**2)
    ini_grid=np.zeros(N**2)
    ini_grid[rand_grid>=frac] = 1
    ini_grid[rand_grid<frac] = -1
    return ini_grid

def m_vector(N, angle, grid):
    m = np.zeros((N**2, 2), dtype=float)
    for i in range(0, N**2):
        s = grid[i]
        m[i] = ( np.column_stack([round(np.cos(angle[i])), np.sin(angle[i])]) )
    return m

#infinite neighborhood
def neighbors(coord, a):
    neighbor_list = []
    sorted_indices = []

    tree = cKDTree(coord)
    distance_threshold = np.sqrt(2) * N + 0.1
    
    for i in range(N**2):
        query_point = coord[i]
        nearest_neighbors_indices = tree.query_ball_point(query_point, distance_threshold)
        nearest_neighbors_indices.remove(i)

        nearest_neighbors_indices = np.insert (nearest_neighbors_indices, 0, i)
        neighbor_list.append(nearest_neighbors_indices)

    neighbor_list = np.array(neighbor_list)
    sorted_indices = np.argsort(neighbor_list[:, 0])
    neighbor_list = neighbor_list[sorted_indices]

    return neighbor_list

@njit
def spin_dipolar_energy(indx, coord, m, neighbor_list):
    e = 0
    alpha = 1
    mi = m[indx]
    for j in range(1,5):
        neigh = neighbor_list[indx, j]
        if neigh != -1:
            r = coord[neigh] - coord[indx]
            dist = np.linalg.norm(r)           
            mj = m[neigh]
            #print(neigh,r)
            #print(dist)
            e_dip_1 = -mj / dist**3
            e_dip_2 = 3 * r * mj.dot(r) / dist**5 
            e_dip = e_dip_1 + e_dip_2

        e += alpha * np.dot(e_dip, mi)
    return e

@njit
def flip(grid, coord, m, neighbor_list, t, N):  
    alpha = 1 
    for k in range(0,N):
        for l in range(0,N):
            indx = np.random.randint(0,N**2) 
            s = grid[indx] 
            mi = s * m[indx]
            s_flip = -1*s  
            mi_flip = s_flip * mi 

            e = 0
            de = 0
            e_flip = 0
            for j in range(1,5):
                neigh = neighbor_list[indx, j]
                if neigh != -1:
                    r = coord[neigh] - coord[indx]
                    dist = np.linalg.norm(r)           
                    mj = m[neigh]

                    e_dip_1 = -mj / dist**3
                    e_dip_2 = 3 * r * mj.dot(r) / dist**5 
                    e_dip = e_dip_1 + e_dip_2

                    e += alpha * np.dot(e_dip, mi)
                    e_flip += alpha * np.dot(e_dip, mi_flip)

                    #de += 2 * ((alpha * 3 * mi.dot(r) * mj.dot(r) )/ dist**5) 
            de = e_flip - e
            if de <= 0:
                s = s_flip
            elif np.random.random() < np.exp(-de/t):
                s = s_flip
        
            grid[indx] = s
    return grid

N = 50
frac = 0.5
a = 1
coord, angle = ising_lattice(N, a)
grid = initial(N, frac)
grid_copy = grid.copy()
neighbor_list = neighbors(coord, a)
m_vectors = m_vector(N,angle,grid)

steps=1000000
eqsteps=1000

'''#index and energy plot
e = []
for indx in range(N**2):
    e.append(spin_dipolar_energy(indx, coord, m_vectors, neighbor_list))
for indx in range(N**2):
    plt.text(coord[indx,0], coord[indx,1], f'{indx, e[indx]}', ha='center', va='center')
plt.scatter(coord[:,0], coord[:,1], c=grid_copy, marker='s')
plt.colorbar()
plt.show()'''

t=0.1
for k in range(eqsteps):
    flip(grid, coord, m_vectors, neighbor_list, t, N)
for l in range(steps):
    flip(grid, coord, m_vectors, neighbor_list, t, N)

plt.figure(figsize=(13,6))
plt.subplot(121)
plt.scatter(coord[:,0], coord[:,1], c=grid_copy, marker='s')
plt.title('Initial config')
plt.colorbar()
plt.subplot(122)
plt.scatter(coord[:,0], coord[:,1], c=grid, marker='s')
plt.colorbar()
plt.title('Final config')
plt.tight_layout()
plt.show()
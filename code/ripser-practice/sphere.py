#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

'''
    Creating some toy data in R^3 to make a sphere
'''

#start of by defining the theta and phi as parameters
parameters = [[np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)] for i in range(1,1000)]

#the coefficients of the sphere equations
a = 1
#the sphere equations using the respective parameters
sphere = [[a*np.cos(par[0]) * np.sin(par[1]), 
          a*np.sin(par[0]) * np.sin(par[1]), 
          a*np.cos(par[1])] for par in parameters]

#now define the data array
data = np.array(sphere, dtype=float)

#plotting our points using ax.scatter because we are in 3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(data[:,0], data[:,1], data[:,2], s=10) 
ax.view_init(azim = 10, elev = 50)
ax.set_xlim3d(-3,3)
ax.set_ylim3d(-3,3)
ax.set_zlim3d(-3,3)

#creating our distance matrix
dist_matrix = sp.spatial.distance.squareform(sp.spatial.distance.pdist(data, metric='euclidean'))

#creating our csv file to later use in live ripser dot org
np.savetxt('sphere.csv', dist_matrix, delimiter=',')
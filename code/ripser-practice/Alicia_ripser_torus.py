#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:49:51 2018

@author: alicia
"""

import numpy as np
import scipy.spatial
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

'''
    Creating some toy data in R^3 to make a torus
'''

#start of by defining the theta and phi as parameters
parameters = [[np.random.uniform(0, 2*np.pi), np.random.uniform(0, 2*np.pi)] for i in range(1,2000)]

#the coefficients of the torus equations
c, a = 2, 1
#the torus equations using the respective parameters
#par[0]=theta
#par[1]=phi
torus = [[(c + a*np.cos(par[0])) * np.cos(par[1]), 
          (c + a*np.cos(par[0])) * np.sin(par[1]), 
          a * np.sin(par[0])] for par in parameters]

#now define the data array
data = np.array(torus, dtype=float)

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
np.savetxt('Alica_ripser_torus.csv', dist_matrix, delimiter=',')
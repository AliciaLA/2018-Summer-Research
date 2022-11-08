import numpy as np
import scipy.spatial
import scipy as sp
import matplotlib.pyplot as plt

'''
    Creating some toy data in R^2.
    Three rings, each sharing a point with the last
'''
ring1 = [[5*np.sin(x),
          5*np.cos(x)] for x in np.arange(0, 2*np.pi, np.pi/10)]

ring2 = [[3*np.sin(x)+8.0,
          3*np.cos(x)] for x in np.arange(0, 2*np.pi, np.pi/12)]

ring3 = [[1*np.sin(x)+12.0,
          1*np.cos(x)] for x in np.arange(0, 2*np.pi, np.pi/4)]


'''
    Arrange the coordinates from the rings in an array.
'''
data = np.array(ring1+ring2+ring3)

'''
    Plot the points so we can see what they look like.
    (s determines the size of the points)
'''
plt.scatter(data[:, 0], data[:, 1], s=10)

'''
    Create a distance matrix.
'''
dist_matrix = sp.spatial.distance.squareform(
        sp.spatial.distance.pdist(data, metric='euclidean'))

'''
    Create a .csv file to upload to ripser
'''
np.savetxt('three_rings.csv', dist_matrix, delimiter=',')

'''
    Exercise:
    Go to http://live.ripser.org/
    - Load a 'distance matrix' to compute...
    - in dimensions '0' to '2' and up to distance '6':
    - Upload the .csv file that was created

    What does the output mean? Document your observations.
    Create more examples and record your observations.

'''

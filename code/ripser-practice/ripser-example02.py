import numpy as np
import scipy.spatial
import scipy as sp
import matplotlib.pyplot as plt

'''
    Creating some toy data in R^2.
'''
# Chose the lower right hand corners
c1 = [0, 0]
c2 = [5, 0]
c3 = [-2, 4]
c4 = [7, 4]
c5 = [2.5, 7]

# Chose the side lengths
s1 = 1
s2 = 1
s3 = 1
s4 = 1
s5 = 1

# Create squares with the given side lengths at the given corners.
sq1 = [c1, [c1[0]+s1, c1[1]], [c1[0], c1[1]+s1], [c1[0]+s1, c1[1]+s1]]
sq2 = [c2, [c2[0]+s2, c2[1]], [c2[0], c2[1]+s2], [c2[0]+s2, c2[1]+s2]]
sq3 = [c3, [c3[0]+s3, c3[1]], [c3[0], c3[1]+s3], [c3[0]+s3, c3[1]+s3]]
sq4 = [c4, [c4[0]+s4, c4[1]], [c4[0], c4[1]+s4], [c4[0]+s4, c4[1]+s4]]
sq5 = [c5, [c5[0]+s5, c5[1]], [c5[0], c5[1]+s5], [c5[0]+s5, c5[1]+s5]]

'''
    Arrange the coordinates from the squares in an array.
'''
data = np.array(sq1 + sq2 + sq3 + sq4 + sq5)

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
np.savetxt('five-squares.csv', dist_matrix, delimiter=',')

'''
    Exercise:
    Go to http://live.ripser.org/
    - Load a 'lower dist matrix' to compute...
    - in dimensions '0' to '2' and up to distance '8':
    - Upload the .csv file that was created

    What does the output mean? Document your observations.
    Create more examples and record your observations.

'''

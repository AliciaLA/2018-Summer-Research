import kmapper as km
import numpy as np
import sklearn

'''
    Creating some toy data in R^3.
    ring1 = 640 points near a ring of radius 10,
                centered at the origin,
                in the xy-plane
    ring2 = 640 points near a ring of radius 4,
                centered at the (10,0,0),
                in the yz-plane
'''
ring1 = [[10*np.sin(x),
          10*np.cos(x),
          0.1*np.sin(np.random.uniform())] for x in np.arange(0, 64, 0.1)]

ring2 = [[0.05*np.sin(np.random.uniform())+10,
          4*np.sin(x),
          4*np.cos(x)] for x in np.arange(0, 64, 0.1)]

'''
    Arrange the coordinates from the rings into an array,
     and create custom labels.
'''
data = np.array(ring1+ring2)
labels = np.array(['Big Ring']*640 + ['Small Ring']*640).transpose()

'''
    Initialize an instance of KeplerMapper.
'''
mapper = km.KeplerMapper(verbose=1)

'''
    Project the data into the xy-plane.
    Note:
        projection=[0]   projects the data onto the x-axis
        projection=[1]   projects the data onto the y-axis
        projection=[0,2] projects the data onto the xz-plane
'''
lens = mapper.fit_transform(data, projection=[0, 1])

'''
    Apply the MAPPER algorithm to the projected data.
    Cover the projected data with hypercubes using
     km.Cover(n_cubes=15, perc_overlap=0.25).
        "n_cubes" refers to the number of cubes in each dimension
        "perc_overlap" refers to how much the bins overlap one another
         as a percent.
        Here, we essentially covering the region of the xy-plane in which
         our data lies by a 15x15 grid of 225 overlapping rectangles.
    Cluster in each bin using sklearn.cluster.DBSCAN(eps=0.5, min_samples=5).
        "eps" refers to the maximum euclidean distance between two points
         for them to be considered in the same cluster.
        "min_samples" refers to the minimum number of points that must be
         withinin eps of eachother before clustering them all together.
'''
simplicial_complex = mapper.map(
        lens,
        X=data,
        clusterer=sklearn.cluster.DBSCAN(eps=0.5, min_samples=5),
        cover=km.Cover(n_cubes=15, perc_overlap=0.25))

'''
    Generate the visualization of the simplicial complex we've just built.
'''
mapper.visualize(simplicial_complex,
                 path_html="keplermapper-toy-rings.html",
                 custom_meta={'Data': "toy rings"},
                 custom_tooltips=labels)

'''
    Exercise:
    - Create your own toy examples with (slightly )more complicated structures.
    - Expiriment with the choices of projection, custering, and covering.
    - Document your observations.

'''

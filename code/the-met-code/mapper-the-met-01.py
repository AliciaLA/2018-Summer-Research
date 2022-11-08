''' Create a distance matrix using mydist '''
dist_matrix = sp.spatial.distance.squareform(sp.spatial.distance.pdist(sub, metric=mydist))
np.savetxt('1000pcs.csv', dist_matrix, delimiter=',')

''' Initialize Mapper '''
mapper: KeplerMapper = kmapper.KeplerMapper(verbose=2)

''' Project onto Dimensions '''
lens = mapper.fit_transform(sub, projection=[5], scaler=None)

''' Create 500 hypercubes with 50% overlapping using DBSCAN clustering algorithm with mydist '''
simplicial_complex: dict = mapper.map(lens,
                                sub,
                                nr_cubes=500,
                                overlap_perc=0.5, clusterer=sklearn.cluster.DBSCAN(eps=3, metric=mydist, algorithm='brute'))
''' Color the graph according to IsHighlight'''
my_colors = np.array([0 if x == 0 else 1 for x in pd.DataFrame(sub)[0]])
''' Generate a webpage '''
html = mapper.visualize(simplicial_complex, path_html="art.html", color_function=my_colors)
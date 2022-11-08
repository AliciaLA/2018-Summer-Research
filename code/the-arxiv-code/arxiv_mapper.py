import pandas as pd
import numpy as np
import kmapper as km
from sklearn import *

def allnonzero(lst):
	return reduce(lambda x, y: (x != 0) and (y != 0), lst)   

def main():
	df = pd.read_csv('arxiv_data.csv', index_col = False)
	numerical_df = df[['weights', 'pages', 'references']].values
	numerical_df = filter(lambda x: allnonzero(x), numerical_df)
	X = np.asarray(numerical_df)
	
	mapper = km.KeplerMapper(verbose=1)
	
	model = ensemble.IsolationForest()
	model.fit(X)
	
	lens1 = model.decision_function(X).reshape(X.shape[0], 1)
	lens2 = mapper.fit_transform(X, projection="L2norm")
	lens3 = mapper.fit_transform(X, projection=[0])
	
	lens = np.c_[lens1, lens2, lens3]
	
	simplicial_complex = mapper.map(lens, X, cover = km.Cover(n_cubes=15, perc_overlap = 0.7), clusterer = cluster.KMeans(n_clusters = 2))
	
	mapper.visualize(simplicial_complex, path_html = "output.html", title = "arxiv data")

main()

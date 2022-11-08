import pandas as pd
import numpy as np
import kmapper
import sklearn
from kmapper import KeplerMapper
import re
import scipy as sp
import string
import functools
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

#Clean data
df = pd.read_csv('PoPCites.csv', encoding="ISO-8859-1")
feature_names = ['Cites', 'Authors', 'Title', 'Age', 'GSRank', 'CitesPerYear', 'CitesPerAuthor', 'AuthorCount', 'Publisher']
sub = df[feature_names].dropna(axis=0)
mydict_authors={}
mydict_title={}
sub['Authors'] = sub['Authors'].astype('category')
sub['Title'] = sub['Title'].astype('category')
sub['Publisher']=sub['Publisher'].astype('category')
sub['author_cat'] = sub['Authors'].cat.codes
sub['title_cat'] = sub['Title'].cat.codes
sub['pub_cat']=sub['Publisher'].cat.codes
for idx, item in enumerate(sub['title_cat']):
    title_words = ' '.join(word.strip(string.punctuation) for word in sub['Title'][idx].split()).lower().split()
    for word in title_words:  # iterating on a copy since removing will mess things up
        if word in stop_words:
            title_words.remove(word)
    title_words = [porter.stem(word) for word in title_words]
    mydict_title[item]=title_words

for idx, item in enumerate(sub['author_cat']):
    mydict_authors[item]=sub['Authors'][idx].strip('…').split(', ')

sub = sub.drop(columns=['Authors', 'Title', 'Publisher'])
# Metric: Common authors, title nltk, publisher
def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def cat_dist(x, y):
    return 1-jaccard(x, y)

def mydist(x,y):
    x_authors=set(mydict_authors.get(x[6]))
    y_authors=set(mydict_authors.get(y[6]))
    x_title=set(mydict_title.get(x[7]))
    y_title=set(mydict_title.get(y[7]))
    author_distance=cat_dist(x_authors,y_authors)
    title_distance=cat_dist(x_title,y_title)
    publisher_distance=1
    if x[8] == y[8]:
        publisher_distance=0
    return author_distance+title_distance+publisher_distance

my_colors=np.array(sub['Cites'])

'''Initialize Mapper'''
mapper: KeplerMapper = kmapper.KeplerMapper(verbose=2)
'''Initialize MDS, our lens function for Mapper.
   Since we are using mydist, a custom metric, we set dissmilarity. '''
mds=sklearn.manifold.MDS(dissimilarity='precomputed')
'''Using mydist, compute a distance matrix for our sample and save it.'''
dist_matrix = sp.spatial.distance.squareform(sp.spatial.distance.pdist(sub, metric=mydist))
np.savetxt('papers.csv', dist_matrix, delimiter=',')
'''Apply MDS to our sample to get our lens.'''
lens=mds.fit_transform(dist_matrix)
'''Using DBSCAN and our metric, generate simplicial complex.'''
simplicial_complex= mapper.map(lens,
                                sub,
                                nr_cubes=10,
                                overlap_perc=0.5,
                                clusterer=sklearn.cluster.DBSCAN(eps=1.75, metric=mydist, algorithm='brute', min_samples=3))
''' Output the simplicial complex, colored by 'Cites', as a webpage. '''
html = mapper.visualize(simplicial_complex, path_html="papers_colored_by_cites.html", color_function=my_colors)

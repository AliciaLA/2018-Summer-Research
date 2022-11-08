'''
Origial source: kepler-mapper/examples/breast-cancer/breast-cancer.py
Repositiory: https://github.com/MLWave/kepler-mapper
Accessed on: 2018/05/31
Modifications by Marcos Ortiz: https://github.com/marcoswastaken

Licence from source, without modification:

The MIT License (MIT)

Copyright (c) 2015 Triskelion - HJ van Veen - info@mlwave.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
'''

import pandas as pd
import numpy as np
import kmapper as km
import sklearn
from sklearn import ensemble

'''
    Import the  Wisconsin Breast Cancer Dataset from:
        https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

    Use pandas to read the csv values into a dataframe
'''

df = pd.read_csv("data.csv")

'''
    Create a list of feature_names to streamline access to those columns in
     the dataframe

    Recode the "diagnosis" feature as 1 for "M" and 0 otherwise.

    Create an array from the features columns of the dataframe. I.e. the
     columns other than "id" and "diagnosis". The "id" is diagnostically
     meaningless, and we hope to evaluate our model based on how well it
     allows us to find connections and differences between samples with
     similar and different diagnoses. (Fill missing entries with zeros.)
'''

feature_names = [c for c in df.columns if c not in ["id", "diagnosis"]]

df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)

X = np.array(df[feature_names].fillna(0))

'''
    Various lenses can be applied to the data for use with the MAPPER
     Algorithm.
    Each lens provides an interval which can be covered by overlapping
     subitervals.
    We can combine these lenses, projecting our data into as many dimensions
     as lenses, and covering with hypercube generalizations of the overlapping
     subitervals in each. For example, if we combine two lenses, then we are
     projecting the data into a region of the form I x J, where I is the
     interval in which the projected data lies in the first lens, and J is
     the interval in which the projected data lies in the second lens. Then, we
     cover I with n_cubes intervals of equal length, i_1, i_2, ..., i_{n-cubes}
     and cover J similarly with intervals j_1, j_2, ..., j_{n-cubes}. In the
     product, I x J, the subset i_r x j_s is a rectangle, and the collection
     of all such rectangles is our cover by overlapping hypercubes in dimension
     2.
'''
# Initialize a mapper
mapper = km.KeplerMapper(verbose=1)

# A custom 1-D lens with Isolation Forest
model = ensemble.IsolationForest(random_state=1729)
model.fit(X)
lens1 = model.decision_function(X).reshape((X.shape[0], 1))

# A 1-D lens with L2-norm
lens2 = mapper.fit_transform(X, projection="l2norm")

# A 1-D lens projecting the data onto the 1st feature, 'radius_mean'
lens3 = mapper.fit_transform(X, projection=[0])

# Combining lens1 and lens2 to create a 2-D [Isolation Forest, L^2-Norm] lens
lens = np.c_[lens1, lens2]

'''
    Note, you could use any of the lenses above individually, or in any
     combination:
         lens = lens1
         lens = lens2
         lens = lens3
         lens = np.c_[lens1, lens2]
         lens = np.c_[lens1, lens3]
         lens = np.c_[lens2, lens3]
         lens = np.c_[lens1, lens2, lens3]
'''

'''
    Now, our mapper builds a simplicial complex using the MAPPER algorithm.
    1) We hand it our lens and our original data
    2) It covers our lens with hypercubes, so that each sub-lens is covered by
        n_cubes intervals that overlap by perc_overlap percent. An the product
        of each combination of these intevals is a hypercube in our lens space.
    3) In each of these hypercubes, we look at the points in the original data
        that were projected into that particular hypercube. Then, we cluster
        those points (in the original space) using the given clusing algorithm.
        Then, if a cluter in one hypercube has points in common with a clutser
        in another hypercube, we connect those clusters by an edge. In the end
        we have a simplicial complex where each cluster is a node, with edges
        (hopefully) connecting clusters that are related to one another.
'''
simplicial_complex = mapper.map(lens,
                                X,
                                cover=km.Cover(n_cubes=15, perc_overlap=0.7),
                                clusterer=sklearn.cluster.KMeans(
                                        n_clusters=2, random_state=1618033))

'''
    Create a color_function to emphasize the diagnosis of the points in each
     node. Red nodes are associated with diagnosis = 1, blue otherwise.
     our mapper averages the values for nodes that contain members with
     different diagnoses.

    Create an array of diagnoses so that we can identify which dianoses are
     found in particular nodes and clusters in the visualization.

    Generate the visualization of the generated graph.
'''
my_colors = np.array([0 if x == 0 else 1 for x in df["diagnosis"]])
y = np.array(df["diagnosis"])

mapper.visualize(simplicial_complex,
                 path_html="breast-cancer_modified.html",
                 title="Wisconsin Breast Cancer Dataset",
                 custom_tooltips=y,
                 color_function=my_colors)

import pandas as pd
import numpy as np
import dionysus as dio
import scipy.spatial


def main():
    processed_df = pd.read_csv('arxiv_data.csv', index_col=False)
    numerical_df = processed_df[['weights', 'pages', 'references']].values

    numerical_df = filter(lambda x: allnonzero(x), numerical_df)
    numerical_df = np.asarray(numerical_df)

    # import custom metric script to add on more metrics
    metrics = ['euclidean', 'cityblock', 'chebyshev']

# uncomment to see diagrams
'''
    for i in metrics:
        diagrams(numerical_df, i)
'''


def allnonzero(lst):
    return reduce(lambda x, y: (x != 0) and (y != 0), lst)


def diagrams(df, metric_key):
    dist_mat = scipy.spatial.distance.pdist(df, metric=metric_key)
    filt = dio.fill_rips(dist_mat, 2, 30)
    pers = dio.homology_persistence(filt)
    diagram_info = dio.init_diagrams(pers, filt)
    for i in range(len(diagram_info)):
        title = metric_key + ", Dimension " + str(i)
        try:
            print("showing " + title)
            dio.plot.plot_diagram(diagram_info[i],
                                  show=True)
            dio.plot.plot_bars(diagram_info[i],
                               show=True)
        except ValueError:
            print("No Diagram Available with metric: " +
                  str(metric_key) +
                  ", and dimension: " +
                  str(i))


main()

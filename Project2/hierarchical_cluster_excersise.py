#Use a hierarchical cluster algorithm to guess a likely number of cluster present in the data
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 20)

# REFERENCE POINT FROM CLASS EXERCISES
# def classes_exercise():
#     np.random.seed(1)
#     variables = ['Age', 'GPA', 'Income', 'Height']
#     labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5']
#     X = np.random.random_sample([6,4])*10
#     df = pd.DataFrame(X, columns=variables,index=labels)
#     row_dist = pd.DataFrame(squareform(pdist(df,metric='euclidean')),
#                             columns = labels,
#                             index = labels)
#     row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
#
#     row_clusters_labeled = pd.DataFrame(row_clusters,
#                  columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
#                  index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])])
#     row_dendogram = dendrogram(row_clusters, labels = labels)
#
#     print(df)
#     print(row_dist)
#     print("row clusters")
#     print(row_clusters)
#     print("row clusters labeled")
#     print(row_clusters_labeled)
#     plt.tight_layout()
#     plt.ylabel("Distance ")
#     plt.show()
#
#     return None

def assignment2_point2():

    path = r'./data/wines_properties.csv'
    wineData = pd.read_csv(path, skiprows=0)
    clusters = linkage(pdist(wineData, metric='euclidean'), method='complete')
    headers = list(wineData)
    labels = ['row label 1', 'row label 2', 'distance', 'no. of items in clust.']
    print(headers)
    clustersLabeled = pd.DataFrame(clusters,
                                   columns=labels,
                                   index=['cluster %d' % (i + 1) for i in range(clusters.shape[0])]
                                   )
    print(clustersLabeled)
    clusterDendogram = dendrogram(clusters)
    plt.tight_layout()
    plt.ylabel("Distance ")
    plt.show()

    return None

#classes_exercise()
assignment2_point2()

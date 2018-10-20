#Use a hierarchical cluster algorithm to guess a likely number of cluster present in the data
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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

def quantify_data(data, standardization):
    result = data.values
    if standardization:
        result = StandardScaler().fit_transform(result)
    return result



def assignment2_point2():
 
    path = r'./data/wines_properties.csv'
    wine_data = pd.read_csv(path, skiprows=0)
    clusters = linkage(pdist(wine_data, metric='euclidean'), method='complete')
    #headers = list(wine_data)
    #print(headers)
    labels = ['row label 1', 'row label 2', 'distance', 'no. of items in clust.']
    clusters_labeled = pd.DataFrame(clusters,
                                 columns=labels,
                                 index=['cluster %d' % (i + 1) for i in range(clusters.shape[0])]
                                 )
    print(clusters_labeled)
    clusterDendogram = dendrogram(clusters)
    fig = plt.figure()

    plt.tight_layout()
    plt.ylabel("Dist")
    plt.show()

    return None


# with specified PCA Factors Alcohol and Malic_Acid 
def assignment2_point2_specified():

    path = r'./data/wines_properties.csv'
    wine_data = pd.read_csv(path, skiprows=0)
    wine_data_specified_reduced = wine_data.loc[:, ['Alcohol', 'Malic_Acid']]
    wine_data_specified_reduced_matrix = quantify_data(wine_data_specified_reduced, False)
    clusters = linkage(pdist(wine_data_specified_reduced, metric='euclidean'), method='complete')
    #print(wine_data_specified_reduced)
    # headers = list(wine_data)
    labels = ['row label 1', 'row label 2', 'distance', 'no. of items in clust.']
    clusters_labeled = pd.DataFrame(clusters,
                                    columns=labels,
                                    index=['cluster %d' % (i + 1) for i in range(clusters.shape[0])]
                                    )
    #print(clusters_labeled)
    clusterDendogram = dendrogram(clusters,
                                  color_threshold=3.8,
                                  truncate_mode ="lastp",
                                  no_labels = True
                                  )
    for i, d in zip(clusterDendogram['icoord'], clusterDendogram['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            plt.plot(x, y, 'ro')
            plt.annotate("%.3g" % y, (x, y), xytext=(0, -8),
                         textcoords='offset points',
                         va='top', ha='center')
    fig = plt.figure()
    print(clusters_labeled)
    clusterDendogram = dendrogram(clusters)

    plt.tight_layout()
    #plt.ylabel("Dist")
    plt.show()
    return None

assignment2_point2_specified ()

#assignment2_point2()


import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 20)

# Exercise 2 - Hierachical Cluster Analysis
def hierarchical_cluster_analysis():
 
    path = r'./data/wines_properties.csv'
    wine_data = pd.read_csv(path, skiprows=0)
    clusters = linkage(pdist(wine_data, metric='euclidean'), method='complete')
    labels = ['row label 1', 'row label 2', 'distance', 'no. of items in clust.']
    clusters_labeled = pd.DataFrame(clusters,
                                 columns=labels,
                                 index=['cluster %d' % (i + 1) for i in range(clusters.shape[0])]
                                 )
    print(clusters_labeled)
    clusterDendogram = dendrogram(clusters,
                                  color_threshold=712,
                                  no_labels = True
                                  )
    plt.title ("Hierachical Analysis including 14 dimensions")
    plt.xlabel("Cluster Labels excluded for better comprehension")
    plt.ylabel("Dist")
    fig = plt.figure()
    plt.tight_layout()
    plt.show()

    return None
#Use the previous number of cluster to perform a K-means cluster analysis
#• Analyse the “silhouette” of the clusters
#• Plot on the space of the first two dimensions of the PCA the clusters obtained with K-means, using a different
#colour for each cluster.
#• For each cluster, which “original” variables (ex ante the PCA) are more important? Consider the barycenter of
#each cluster (the barycenter is an observation) and its variables values.
#• Using both the information of barycenters and of PCA, give an interpretation to each cluster.
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm
pd.set_option('display.max_columns', 20)



def assignment2_point3():
    path = r'./data/wines_properties.csv'
    wineData = pd.read_csv(path, skiprows=0)
    #use linkage on the wine data with only 2 principal components
    #naive attempt
    wineDataReduced = wineData.loc[:,['Alcohol','Ash']]
    wineDataReducedasMatrix = pd.DataFrame.as_matrix(wineDataReduced)
    print(wineDataReduced)
    clusters = linkage(pdist(wineData, metric='euclidean'), method='complete')
    headers = list(wineData)
    NumberOfClusters = 8
    KmeansCheck = KMeans(n_clusters=NumberOfClusters,
                        init='random')
    wineKmeansCheck = KmeansCheck.fit_predict(wineDataReduced)
    colors = cm.rainbow(np.linspace(0, 1, NumberOfClusters+1))

    print(wineKmeansCheck)
    print(wineDataReducedasMatrix[:,0])
    for i in range(0,NumberOfClusters):
        plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == i, 0],
                    wineDataReducedasMatrix[wineKmeansCheck == i, 1],
                    s=50, c=colors[i],
                    marker='o', edgecolor='black',
                    label='cluster %d' % (i+1))
    plt.scatter(KmeansCheck.cluster_centers_[:, 0],
                KmeansCheck.cluster_centers_[:, 1],
                s=250, marker='*',
                c=colors[NumberOfClusters], edgecolor='black',
                label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()


    return None

assignment2_point3()
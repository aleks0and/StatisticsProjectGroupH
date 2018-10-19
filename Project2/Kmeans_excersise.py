#Use the previous number of cluster to perform a K-means cluster analysis
#• Analyse the “silhouette” of the clusters
#• Plot on the space of the first two dimensions of the PCA the clusters obtained with K-means, using a different
#colour for each cluster.
#• For each cluster, which “original” variables (ex ante the PCA) are more important? Consider the barycenter of
#each cluster (the barycenter is an observation) and its variables values.
#• Using both the information of barycenters and of PCA, give an interpretation to each cluster.
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
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
    KmeansCheck = KMeans(n_clusters=8,
                        init='random')
    wineKmeansCheck = KmeansCheck.fit_predict(wineDataReduced)


    print(wineKmeansCheck)
    print(wineDataReducedasMatrix[:,0])

    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 0, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 0, 1],
                s=50, c='lightgreen',
                marker='s', edgecolor='black',
                label='cluster 1')
    # cluster 2
    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 1, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 1, 1],
                s=50, c='orange',
                marker='o', edgecolor='black',
                label='cluster 2')
    # cluster 3
    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 2, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 2, 1],
                s=50, c='lightblue',
                marker='v', edgecolor='black',
                label='cluster 3')
    # cluster 4
    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 3, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 3, 1],
                s=50, c='red',
                marker='v', edgecolor='black',
                label='cluster 4')
    # cluster 5
    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 4, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 4, 1],
                s=50, c='yellow',
                marker='v', edgecolor='black',
                label='cluster 5')
    # cluster 6
    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 5, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 5, 1],
                s=50, c='brown',
                marker='v', edgecolor='black',
                label='cluster 6')
    # cluster 7
    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 6, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 6, 1],
                s=50, c='purple',
                marker='v', edgecolor='black',
                label='cluster 7')
    # cluster 8
    plt.scatter(wineDataReducedasMatrix[wineKmeansCheck == 7, 0],
                wineDataReducedasMatrix[wineKmeansCheck == 7, 1],
                s=50, c='fuchsia',
                marker='v', edgecolor='black',
                label='cluster 8')
    plt.scatter(KmeansCheck.cluster_centers_[:, 0],
                KmeansCheck.cluster_centers_[:, 1],
                s=250, marker='*',
                c='red', edgecolor='black',
                label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()


    return None

assignment2_point3()
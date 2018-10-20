#Use the previous number of cluster to perform a K-means cluster analysis
#• Analyse the “silhouette” of the clusters
#• Plot on the space of the first two dimensions of the PCA the clusters obtained with K-means, using a different
#colour for each cluster.
#• For each cluster, which “original” variables (ex ante the PCA) are more important? Consider the barycenter of
#each cluster (the barycenter is an observation) and its variables values.
#• Using both the information of barycenters and of PCA, give an interpretation to each cluster.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from matplotlib import cm
from PCA_exercise import pca_top2_extraction
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', 20)


def plot_clusters(data, predicted_clusters, initialized_kmeans, number_of_clusters):
    for i in range(0, number_of_clusters):
        color = cm.nipy_spectral(float(i) / number_of_clusters)
        plt.scatter(data[predicted_clusters == i, 0],
                    data[predicted_clusters == i, 1],
                    s=50, c=color,
                    marker='o', edgecolor=color,
                    label='cluster %d' % (i+1))
    color = cm.nipy_spectral(float(number_of_clusters) / number_of_clusters)
    plt.scatter(initialized_kmeans.cluster_centers_[:, 0],
                initialized_kmeans.cluster_centers_[:,  1],
                s=250, marker='*',
                c=color, edgecolor='black',
                label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()

# loading data and omiting specified number of rows, also dropping the rows with missing values.
def prepare_and_load_data(path,skiprows):

    data = pd.read_csv(path,skiprows=skiprows)
    data.dropna(how="all", inplace=True)
    return data


def quantify_data(data, standardization):
    result = pd.DataFrame.as_matrix(data)
    if standardization:
        result = StandardScaler().fit_transform(result)
    return result


def assignment2_point3():
    path = r'./data/wines_properties.csv'
    wine_data = prepare_and_load_data(path, skiprows=0)
    # PCA missing

    # naive attempt

    wine_data_reduced = wine_data.loc[:, ['Alcohol', 'Ash']]
    wine_data_reduced_matrix = quantify_data(wine_data_reduced, False)
    print(wine_data_reduced)
    headers = list(wine_data)
    print(headers)
    number_of_clusters = 8
    kmeans_init = KMeans(n_clusters=number_of_clusters,
                         init='random')
    wine_predicted_clusters = kmeans_init.fit_predict(wine_data_reduced)
    plot_clusters(wine_data_reduced_matrix, wine_predicted_clusters, kmeans_init, number_of_clusters)
    return None


# assignment2_point3()
# Exercise 4
# based on http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# nice example for the use of this method!
# for the purpose of testing I will not load the database in the arguments.

def best_k_for_kmeans():
    path = r'./data/wines_properties.csv'
    wine_data = prepare_and_load_data(path,skiprows=0)
    # hardcoded column names to be changed for the PCI analysis
    first_two_principal_components = pca_top2_extraction(wine_data)
    print(first_two_principal_components)
    wine_data_reduced = quantify_data(wine_data, True).dot(first_two_principal_components)
    wine_data_reduced_matrix = wine_data_reduced
    # check with pca
    sklearn_pca = PCA(n_components=2)
    wine_data_standardized = quantify_data(wine_data,True)
    Y_sklearn = sklearn_pca.fit_transform(wine_data_standardized)
    # preimplemented part
    #wine_data_reduced_matrix = Y_sklearn
    # range of clusters is now hardcoded but we can get it from hierarchical cluster analysis
    min_cluster = 2
    max_cluster = 11
    silhouette_list = []
    last_best_silhouette_avg = -1
    last_best_cluster_index = 0
    range_cluster = [i for i in range(min_cluster, max_cluster)]
    for clusterI in range_cluster:
        kmeans_setup = KMeans(n_clusters=clusterI, random_state=10)
        predicted_clusters = kmeans_setup.fit_predict(wine_data_reduced_matrix)
        silhouette_avg = silhouette_score(wine_data_reduced_matrix, predicted_clusters)
        silhouette_list.append(silhouette_avg)
        if silhouette_avg > last_best_silhouette_avg:
            last_best_silhouette_avg = silhouette_avg
            last_best_cluster_index = clusterI

    kmeans_setup = KMeans(n_clusters=last_best_cluster_index, random_state=10)
    predicted_clusters = kmeans_setup.fit_predict(wine_data_reduced_matrix)
    print("best silhouette is for %d clusters" % last_best_cluster_index)
    print("the value for best silhouette is: " + str(last_best_silhouette_avg))
    print(silhouette_list)
    plot_clusters(wine_data_reduced_matrix, predicted_clusters, kmeans_setup, last_best_cluster_index)
    return last_best_cluster_index

#assignment2_point3()
best_k_for_kmeans()

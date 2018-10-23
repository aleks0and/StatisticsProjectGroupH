from util import prepare_and_load_data, quantify_data, plot_clusters
from PCA_exercise import pca_top2_extraction
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def best_k_for_kmeans_given_data(data):
    first_two_principal_components = pca_top2_extraction(data)
    wine_data_reduced = quantify_data(data, True).dot(first_two_principal_components)
    wine_data_reduced_matrix = wine_data_reduced
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


def best_k_for_kmeans():
    path = r'./data/wines_properties.csv'
    wine_data = prepare_and_load_data(path, skip_rows=0)
    first_two_principal_components = pca_top2_extraction(wine_data)
    wine_data_reduced = quantify_data(wine_data, True).dot(first_two_principal_components)
    wine_data_reduced_matrix = wine_data_reduced
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

# path = r'./data/wines_properties.csv'
# wine_data = prepare_and_load_data(path, skip_rows=0)
# number = best_k_for_kmeans_given_data(wine_data)

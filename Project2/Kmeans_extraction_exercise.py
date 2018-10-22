from util import prepare_and_load_data, quantify_data, plot_clusters
from PCA_exercise import pca_top2_extraction
from sklearn.metrics import silhouette_score


from sklearn.cluster import KMeans

# min_cluster and max_cluster are the minimum and maximum number of clusters
# which we are considering as possible numbers
def best_k_for_kmeans_given_data(data, min_cluster, max_cluster):
    first_two_principal_components, first_two_principal_components_names = pca_top2_extraction(data)
    print(first_two_principal_components)
    wine_data_reduced = quantify_data(data, True).dot(first_two_principal_components)
    wine_data_reduced_matrix = wine_data_reduced
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
    # hardcoded column names to be changed for the PCI analysis
    first_two_principal_components, first_two_principal_components_names = pca_top2_extraction(wine_data)
    print(first_two_principal_components)
    wine_data_reduced = quantify_data(wine_data, True).dot(first_two_principal_components)
    wine_data_reduced_matrix = wine_data_reduced
    # check with pca
    #sklearn_pca = PCA(n_components=2)
    #wine_data_standardized = quantify_data(wine_data,True)
    #Y_sklearn = sklearn_pca.fit_transform(wine_data_standardized)
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

path = r'./data/wines_properties.csv'
wine_data = prepare_and_load_data(path, skip_rows=0)
#number = best_k_for_kmeans_given_data(wine_data,2,11)

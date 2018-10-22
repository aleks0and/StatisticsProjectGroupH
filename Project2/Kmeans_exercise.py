import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from util import prepare_and_load_data, plot_clusters, quantify_data
pd.set_option('display.max_columns', 20)


def assignment2_point3():
    
    # Loading the data
    path = r'./data/wines_properties.csv'
    wine_data = prepare_and_load_data(path, skiprows=0)
    wine_data_matrix = quantify_data(wine_data, False)
    
    # Performing Kmeans
    number_of_clusters = 8
    kmeans_init = KMeans(n_clusters=number_of_clusters,
                         init='random')
    wine_predicted_clusters = kmeans_init.fit_predict(wine_data_matrix)
    
    #Plotting of results
    plot_clusters(wine_data_matrix, wine_predicted_clusters, kmeans_init, number_of_clusters)
    return None



##### 3.1. Creating the silhouette plot #####

def silhouette():
    
    #Processing the data
    path = r'./data/wines_properties.csv'
    wine_data = prepare_and_load_data(path, skiprows=0)
    
    #Performing Kmeans
    km = KMeans(n_clusters=3, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
    y_km = km.fit_predict(wine_data)
    cluster_labels = np.unique(y_km)
    
    n_clusters = cluster_labels.shape[0]
    
    #Creating the silhouette plot
    silhouette_vals = silhouette_samples(wine_data, y_km, metric='euclidean')
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                 edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_vals)
    
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 

    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')

    plt.tight_layout()
    plt.show()
    
#Exercise 3.2.

def assignment2_point3_top2_eigenvalues():
    
    #Data processing
    path = r'./data/wines_properties.csv'
    wine_data = prepare_and_load_data(path, skiprows=0)
    wine_data_reduced = wine_data.loc[:, ['Alcohol', 'Malic_Acid']]
    wine_data_reduced_matrix = quantify_data(wine_data_reduced, False)
    
    #Performing Kmeans
    number_of_clusters = 3
    kmeans_init = KMeans(n_clusters=number_of_clusters,
                         init='random')
    wine_predicted_clusters = kmeans_init.fit_predict(wine_data_reduced)
    plot_clusters(wine_data_reduced_matrix, wine_predicted_clusters, kmeans_init, number_of_clusters)
    print("plotted assignment2_point3")
    return None
# assignment2_point3_top2_eigenvalues()

    
# Execrise 3.3. + 3.4.
    
def original_vars_PCA():
    path = r'./data/wines_properties.csv'
    wine_data = prepare_and_load_data(path, skip_rows=0)
    wine_data.dropna(how="all", inplace=True)
    km = KMeans(n_clusters=3, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
    y_km = km.fit_predict(wine_data)
    
    
    names = ["Alcohol","Malic_Acid","Ash","Ash_Alcanity","Magnesium","Total_Phenols",
             "Flavanoids","Nonflavanoid_Phenols","Proanthocyanins",
             "Color_Intensity","Hue","OD280","Proline","Cluster"]

    wine_data_with_clusters = pd.DataFrame(np.hstack((wine_data, y_km.reshape(-1, 1))), columns = names)
    
    
    # Defining three cluster for PCA purpose
    cluster1 = wine_data_with_clusters[wine_data_with_clusters["Cluster"]  == 0]
    cluster2 = wine_data_with_clusters[wine_data_with_clusters["Cluster"]  == 1]
    cluster3 = wine_data_with_clusters[wine_data_with_clusters["Cluster"]  == 2]

    
    # PCA for the first cluster
    x_s = quantify_data(cluster1, True)
    corelation_matrix = np.corrcoef(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(corelation_matrix[:-1,:-1])
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i], names[i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    top2_eigenvectors = np.hstack((eigen_pairs[0][1].reshape(len(eigen_values), 1),
                                   eigen_pairs[1][1].reshape(len(eigen_values), 1)))
    
    top2_withnames = pd.DataFrame(top2_eigenvectors, columns = [eigen_pairs[0][2],eigen_pairs[1][2]])
    print(top2_withnames)
    return top2_withnames


    # PCA for the second cluster
    x_s = quantify_data(cluster2, True)
    corelation_matrix = np.corrcoef(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(corelation_matrix[:-1,:-1])
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i], names[i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    top2_eigenvectors = np.hstack((eigen_pairs[0][1].reshape(len(eigen_values), 1),
                                   eigen_pairs[1][1].reshape(len(eigen_values), 1)))
    
    top2_withnames = pd.DataFrame(top2_eigenvectors, columns = [eigen_pairs[0][2],eigen_pairs[1][2]])
    print(top2_withnames)
    return top2_withnames

  # PCA for the third cluster
    x_s = quantify_data(cluster3, True)
    corelation_matrix = np.corrcoef(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(corelation_matrix[:-1,:-1])
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i], names[i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    top2_eigenvectors = np.hstack((eigen_pairs[0][1].reshape(len(eigen_values), 1),
                                   eigen_pairs[1][1].reshape(len(eigen_values), 1)))
    
    top2_withnames = pd.DataFrame(top2_eigenvectors, columns = [eigen_pairs[0][2],eigen_pairs[1][2]])
    print(top2_withnames)
    return top2_withnames

original_vars_PCA()
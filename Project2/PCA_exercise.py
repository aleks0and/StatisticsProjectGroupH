import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
from util import prepare_and_load_data, quantify_data, plot_clusters
import matplotlib.cm as cm




def pca_top2_extraction(data):
    names = list(data)
    x_s = quantify_data(data, True)
    corelation_matrix = np.corrcoef(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(corelation_matrix)
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i], names[i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    top2_eigenvectors = np.hstack((eigen_pairs[0][1].reshape(len(eigen_values), 1),
                                   eigen_pairs[1][1].reshape(len(eigen_values), 1)))
    
    top2_withnames = pd.DataFrame(top2_eigenvectors, columns=[eigen_pairs[0][2], eigen_pairs[1][2]])
    return top2_withnames

def pca_top2_extraction_testing(data):
    x_s = quantify_data(data, True)
    names = ["Alcohol", "Malic_Acid", "Ash", "Ash_Alcanity",
             "Magnesium", "Total_Phenols", "Flavanoids",
             "Nonflavanoid_Phenols", "Proanthocyanins",
             "Color_Intensity", "Hue", "OD280", "Proline",
             "Customer_Segment"]
    corelation_matrix = np.corrcoef(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(corelation_matrix)
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i], names[i]) for i in range(len(eigen_values))]
    eigen_unsorted = eigen_pairs
    print(eigen_pairs, eigen_unsorted)
    eigen_pairs.sort()
    eigen_pairs.reverse()
    top2_eigenvectors = np.hstack((eigen_pairs[0][1].reshape(len(eigen_values), 1),
                                   eigen_pairs[1][1].reshape(len(eigen_values), 1)))

    top2_withnames = pd.DataFrame(top2_eigenvectors, columns=[eigen_pairs[0][2], eigen_pairs[1][2]])
    return top2_withnames

def pca_exercise():
    path = r'./data/wines_properties.csv'
    data = prepare_and_load_data(path ,skip_rows = 0)
    x_s = quantify_data(data, True)
    covariance_matrix = np.cov(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]
    eigen_vectors_values.sort()
    eigen_vectors_values.reverse()
    top2_eigenvectors = np.hstack((eigen_vectors_values[0][1].reshape(len(eigen_values), 1),
                                   eigen_vectors_values[1][1].reshape(len(eigen_values), 1),
                                   ))
    py.offline.init_notebook_mode(connected=True)
    PCs = eigen_vectors
    colors = cm.rainbow(np.linspace(0, 1, len(PCs)))
    fig = plt.figure(figsize=(5, 5))
    plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
               PCs[0, ], PCs[1, :],
               angles='xy', scale_units='xy', scale=1, color=colors)
    feature_names = list(i for i in range(0,14))
    for i, j, z in zip(PCs[1, :]+0.02, PCs[0, :]+0.02, feature_names):
        plt.text(j, i, z, ha='center', va='center')
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
    plt.gca().add_artist(circle)

    plt.axis('equal')
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.xlabel('PC 0')
    plt.ylabel('PC 1')

    plt.show()


def scree_plot():
    # Scree plot for our report
    # Preparing the data
    data = prepare_and_load_data(path = r'./data/wines_properties.csv', skip_rows = 0)
    X_s = quantify_data(data, True)
    mean_vector = np.mean(X_s, axis = 0)
    N = X_s.shape[0]
    # Generating the Covariance Matrix, Eigenvectors and Eigenvalues
    covariance_matrix = (X_s - mean_vector).T.dot((X_s - mean_vector)) / (N-1)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    pd.DataFrame(eigen_values)
    # Generating data for the Scree Plot
    tot_eig_vals = sum (eigen_values)
    sorted_eigenvalues = sorted(eigen_values, reverse = True )
    variance_explained = [(i / tot_eig_vals)*100 for i in sorted_eigenvalues]
    # Plotting the Scree Plot
    plt.xlabel("Dimensions")
    plt.plot(1/np.cumsum(variance_explained))

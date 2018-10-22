import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from util import prepare_and_load_data, quantify_data
from sklearn.decomposition import PCA


def principal_components_comparison():
    path = r'./data/wines_properties.csv'
    data = prepare_and_load_data(path,skip_rows=0)
    x_s = quantify_data(data, True)
    covariance_matrix = np.cov(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]
    # eigen_vectors_values.sort()
    # eigen_vectors_values.reverse()
    top2_eigenvectors = np.hstack((eigen_vectors_values[0][1].reshape(len(eigen_values), 1),
                                   eigen_vectors_values[1][1].reshape(len(eigen_values), 1),
                                   ))

    py.offline.init_notebook_mode(connected=True)
    feature_names = list(data)
    PCs = eigen_vectors
    colors = cm.rainbow(np.linspace(0, 1, 2))
    compared_vectors_name = []
    plt.figure(figsize=(15,9))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(1,len(PCs)):
        compared_vectors_name.clear()
        compared_vectors_name.append(feature_names[0])
        compared_vectors_name.append(feature_names[i])
        plt.subplot(3,5,i)
        plt.axis('equal')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.xlabel('PC 0')
        plt.ylabel('PC 1')
        plt.title('%d' %i)
        plt.grid(True)
        compared_eigenvectors = np.hstack((eigen_vectors_values[0][1].reshape(len(eigen_values), 1),
                                           eigen_vectors_values[i][1].reshape(len(eigen_values), 1)
                                           ))
        plt.quiver(np.zeros(compared_eigenvectors.shape[1]), np.zeros(compared_eigenvectors.shape[1]),
                   compared_eigenvectors[0,], compared_eigenvectors[1, :],
                   angles='xy', scale_units='xy', scale=1, color=colors)
        for i, j, z in zip(compared_eigenvectors[1, :] + 0.02, compared_eigenvectors[0, :] + 0.02, compared_vectors_name):
            plt.text(j, i, z, ha='center', va='center')
        circle = plt.Circle((0, 0), 0.5, facecolor='none', edgecolor='b')
        plt.gca().add_artist(circle)

    plt.show()
    



principal_components_comparison()
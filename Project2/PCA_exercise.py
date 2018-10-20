import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def prepare_and_load_data(path, skip_rows):
    data = pd.read_csv(path, skiprows=skip_rows)
    data.dropna(how="all", inplace=True)
    return data


def quantify_data(dataset, standardization):
    result = dataset.values
    if standardization:
        result = StandardScaler().fit_transform(result)
    return result


def pca_top2_extraction(data):
    x_s = quantify_data(data, True)  
    names = list(data)
    correlation_matrix = np.corrcoef(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(correlation_matrix)
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i], names[i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    top2_eigenvectors = np.hstack((eigen_pairs[0][1].reshape(len(eigen_values), 1),
                                   eigen_pairs[1][1].reshape(len(eigen_values), 1)))
    
    top2_withnames = pd.DataFrame(top2_eigenvectors, columns = [eigen_pairs[0][2],eigen_pairs[1][2]])
    return top2_withnames

def pca_exercise():
    path = r'./data/wines_properties.csv'
    data = prepare_and_load_data(path,skip_rows=0)
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
    # TESTING
    # plotting all of the components
    # pca_maker = PCA(n_components=2)
    # pca = pca_maker.fit_transform(x_s)
    # PCs = pca_maker.components_
    # for plotting only the two most important ones
    # PCs = PCA().fit(x_s).components_
    # PCs = pca_top2_extraction(data)
    PCs = eigen_vectors
    # PCs = top2_eigenvectors

    plt.figure(figsize=(5, 5))
    plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
               PCs[0, ], PCs[1, :],
               angles='xy', scale_units='xy', scale=1)

    feature_names = list(data)
    for i, j, z in zip(PCs[1, :]+0.02, PCs[0, :]+0.02, feature_names):
        plt.text(j, i, z, ha='center', va='center')

    # Add unit circle
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='b')
    plt.gca().add_artist(circle)

    plt.axis('equal')
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    plt.xlabel('PC 0')
    plt.ylabel('PC 1')

    plt.show()
    




pca_exercise()
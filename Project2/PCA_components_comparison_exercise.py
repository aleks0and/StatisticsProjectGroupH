import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import matplotlib.cm as cm
from util import prepare_and_load_data, quantify_data

def principal_components_comparison_given_columns(column_id):
    path = r'./data/wines_properties.csv'
    data = prepare_and_load_data(path,skip_rows=0)
    x_s = quantify_data(data, True)
    covariance_matrix = np.cov(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]
    py.offline.init_notebook_mode(connected=True)
    feature_names = list(data)
    PCs = eigen_vectors
    colors = cm.rainbow(np.linspace(0, 1, 2))
    compared_vectors_name = []
    plt.figure(figsize=(15,9))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(0,len(PCs)):
        if i != column_id:
            compared_vectors_name.clear()
            compared_vectors_name.append(feature_names[column_id])
            compared_vectors_name.append(feature_names[i])
            plt.subplot(3, 5, i+1)
            plt.axis('equal')
            plt.xlim([-1.0, 1.0])
            plt.ylim([-1.0, 1.0])
            plt.xlabel('PC 0')
            plt.ylabel('PC 1')
            plt.title('%d' %i )
            plt.grid(True)
            compared_eigenvectors = np.hstack((eigen_vectors_values[column_id][1].reshape(len(eigen_values), 1),
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

def principal_components_comparison_given_data(data, column_id):
    x_s = quantify_data(data, True)
    covariance_matrix = np.cov(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]
    py.offline.init_notebook_mode(connected=True)
    PCs = eigen_vectors
    colors = cm.rainbow(np.linspace(0, 1, 2))
    compared_vectors_name = []
    plt.figure(figsize=(15,9))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(0,len(PCs)):
        if i != column_id:
            compared_vectors_name.clear()
            compared_vectors_name.append("PCA " + str(column_id))
            compared_vectors_name.append("PCA " + str(i))
            plt.subplot(3, 5, i+1)
            plt.axis('equal')
            plt.xlim([-1.0, 1.0])
            plt.ylim([-1.0, 1.0])
            plt.xlabel('PC 0')
            plt.ylabel('PC 1')
            plt.title('%d' % i)
            plt.grid(True)
            compared_eigenvectors = np.hstack((eigen_vectors_values[column_id][1].reshape(len(eigen_values), 1),
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

def principal_components_comparison_3by3(data):

    x_s = quantify_data(data, True)
    covariance_matrix = np.cov(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]
    py.offline.init_notebook_mode(connected=True)
    PCs = eigen_vectors
    X_axis = 3
    Y_axis = 3
    plotsize_x = 9
    plotsize_y = 9
    colors = cm.rainbow(np.linspace(0, 1, 2))
    compared_vectors_name = []
    count = len(PCs)*(len(PCs)-1)/2
    current_index = 0
    vectors_to_compare = (len(PCs)-1)
    vectors_compared = (len(PCs)-1)
    for k in range(0, int(count/9)+1):
        plt.figure(figsize=(plotsize_x, plotsize_y))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        for i in range(0,9):
            if (vectors_to_compare == -1):
                current_index += 1
                vectors_compared += -1
                vectors_to_compare = vectors_compared
            if (vectors_compared != 0):
                if current_index == (current_index + (vectors_compared - vectors_to_compare)):
                    vectors_to_compare += -1
                compared_index = (current_index + (vectors_compared - vectors_to_compare))
                compared_vectors_name.clear()
                compared_vectors_name.append("PCA"+str(current_index))
                compared_vectors_name.append("PCA"+str(compared_index))
                plt.subplot(X_axis, Y_axis, i + 1)
                plt.axis('equal')
                plt.xlim([-1.0, 1.0])
                plt.ylim([-1.0, 1.0])
                plt.xlabel('PC %d' % current_index)
                plt.ylabel('PC %d' % (compared_index))
                plt.title('%d' % i)
                plt.grid(True)
                compared_eigenvectors = np.hstack((eigen_vectors_values[current_index][1].reshape(len(eigen_values), 1),
                                                   eigen_vectors_values[compared_index][1].reshape(len(eigen_values), 1)
                                                   ))
                plt.quiver(np.zeros(compared_eigenvectors.shape[1]), np.zeros(compared_eigenvectors.shape[1]),
                           compared_eigenvectors[0,], compared_eigenvectors[1, :],
                           angles='xy', scale_units='xy', scale=1.5, color=colors)
                for i, j, z in zip(compared_eigenvectors[1, :] + 0.02, compared_eigenvectors[0, :] + 0.02,
                                   compared_vectors_name):
                    plt.text(j, i, z, ha='center', va='center')
                circle = plt.Circle((0, 0), 0.5, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)
                vectors_to_compare += -1
    plt.show()

def principal_components_comparison_given_data_all_in_one(data):
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

# Run the comparison between the 3rd PC and all the others
 #principal_components_comparison_given_columns(2)
# Run the comparison between the 4th PC and all the others given dataset.
# we assume that the dataset is prepraed beferehand.
#path = r'./data/wines_properties.csv'
#wine_data = prepare_and_load_data(path, skip_rows=0)
# #principal_components_comparison_given_data(wine_data, 1)
#principal_components_comparison_3by3(wine_data)
# path = r'./data/wines_properties.csv'
# wine_data = prepare_and_load_data(path, skip_rows=0)
# principal_components_comparison_given_data(wine_data,3)

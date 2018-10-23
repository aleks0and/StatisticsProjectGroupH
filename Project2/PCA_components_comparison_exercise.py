import numpy as np
import matplotlib.pyplot as plt
import plotly as py
import matplotlib.cm as cm
from util import prepare_and_load_data, quantify_data

def principal_components_comparison_given_data(data, column_id):
    x_s = quantify_data(data, True)
    covariance_matrix = np.cov(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]
    py.offline.init_notebook_mode(connected=True)
    PCs = eigen_vectors
    colors = cm.rainbow(np.linspace(0, 1, 14))
    compared_vectors_name = []
    plt.figure(figsize=(16,10))
    skipped = 0
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    for i in range(0,len(PCs)):
        if i != column_id:
            compared_vectors_name.clear()
            compared_vectors_name = list(i for i in range(0, 14))
            plt.subplot(3, 5, i+1 - skipped)
            plt.axis('equal')
            plt.xlim([-1.0, 1.0])
            plt.ylim([-1.0, 1.0])
            plt.xlabel('PC %d' % column_id)
            plt.ylabel('PC %d' % i)
            plt.title('%d' % i)
            plt.grid(True)
            # compared_eigenvectors = np.hstack((eigen_vectors_values[column_id][1].reshape(len(eigen_values), 1),
            #                                    eigen_vectors_values[i][1].reshape(len(eigen_values), 1)
            #                                    ))
            compared_eigenvectors = eigen_vectors
            plt.quiver(np.zeros(compared_eigenvectors.shape[1]), np.zeros(compared_eigenvectors.shape[1]),
                       compared_eigenvectors[column_id, :], compared_eigenvectors[i, :],
                       angles='xy', scale_units='xy', scale=1, color=colors)
            for i, j, z in zip(compared_eigenvectors[i, :] + 0.02, compared_eigenvectors[column_id, :] + 0.02, compared_vectors_name):
                plt.text(j, i, z, ha='center', va='center')
            circle = plt.Circle((0, 0), 0.5, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)
        else:
            skipped = 1
    plt.show()

def principal_components_comparison_3by3(data):

    x_s = quantify_data(data, True)
    covariance_matrix = np.cov(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]
    py.offline.init_notebook_mode(connected=True)
    PCs = eigen_vectors_values
    X_axis = 3
    Y_axis = 3
    plotsize_x = 12
    plotsize_y = 12
    colors = cm.rainbow(np.linspace(0, 1, len(PCs)))
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
                compared_vectors_name.append(str(current_index))
                compared_vectors_name.append(str(compared_index))
                compared_vectors_name = list(i for i in range(0, len(PCs)))
                plt.subplot(X_axis, Y_axis, i + 1)
                plt.axis('equal')
                plt.xlim([-1.0, 1.0])
                plt.ylim([-1.0, 1.0])
                plt.xlabel('PC %d' % current_index)
                plt.ylabel('PC %d' % (compared_index))
                plt.title('%d' % i)
                plt.grid(True)
                # compared_eigenvectors = np.hstack((eigen_vectors_values[current_index][1].reshape(len(eigen_values), 1),
                #                                    eigen_vectors_values[compared_index][1].reshape(len(eigen_values), 1)
                #                                    ))
                compared_eigenvectors = eigen_vectors
                plt.quiver(np.zeros(compared_eigenvectors.shape[1]), np.zeros(compared_eigenvectors.shape[1]),
                           compared_eigenvectors[current_index, :], compared_eigenvectors[compared_index, :],
                           angles='xy', scale_units='xy', scale=1.5, color=colors)
                for i, j, z in zip(compared_eigenvectors[compared_index, :] + 0.02,
                                   compared_eigenvectors[current_index, :] + 0.02,
                                   compared_vectors_name):
                    plt.text(j, i, z, ha='center', va='center')
                circle = plt.Circle((0, 0), 0.5, facecolor='none', edgecolor='b')
                plt.gca().add_artist(circle)
                vectors_to_compare += -1
    plt.show()


# Run the comparison between the 4th PC and all the others given dataset.
# we assume that the dataset is prepraed beferehand.
# path = r'./data/wines_properties.csv'
# wine_data = prepare_and_load_data(path, skip_rows=0)
# principal_components_comparison_given_data(wine_data, 3)
# principal_components_comparison_3by3(wine_data)
# path = r'./data/wines_properties.csv'
# wine_data = prepare_and_load_data(path, skip_rows=0)
# principal_components_comparison_given_data(wine_data,3)

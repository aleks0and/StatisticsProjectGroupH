import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly as py
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%matplotlib inline


def prepare_and_load_data(path,skiprows):
    data = pd.read_csv(path,skiprows=skiprows)
    data.dropna(how="all", inplace=True)
    return data


def quantify_data(data, standardization):
    result = data.values
    if standardization:
        result = StandardScaler().fit_transform(result)
    return result


def pca_top2_extraction(data):
<<<<<<< HEAD
    x_s = quantify_data(data, True)
    names = list(data)
=======
    x_s = quantify_data(data, True)  
    
    names = ["Alcohol","Malic_Acid","Ash","Ash_Alcanity",
                             "Magnesium","Total_Phenols","Flavanoids",
                             "Nonflavanoid_Phenols","Proanthocyanins",
                             "Color_Intensity","Hue","OD280","Proline",
                             "Customer_Segment"]
    
>>>>>>> 13ff4771733e8761e05cebf5da923eed35e50bc0
    corelation_matrix = np.corrcoef(x_s.T)
    eigen_values, eigen_vectors = np.linalg.eig(corelation_matrix)
    eigen_pairs = [(np.abs(eigen_values[i]), eigen_vectors[:, i], names[i]) for i in range(len(eigen_values))]
    eigen_pairs.sort()
    eigen_pairs.reverse()
    top2_eigenvectors = np.hstack((eigen_pairs[0][1].reshape(len(eigen_values), 1),
                                   eigen_pairs[1][1].reshape(len(eigen_values), 1)))
    
    top2_withnames = pd.DataFrame(top2_eigenvectors, columns = [eigen_pairs[0][2],eigen_pairs[1][2]])
    
    return top2_withnames

def pca_exercise():
    path = r'./data/wines_properties.csv'
    data = prepare_and_load_data(path,skiprows=0)
    # Storing only the numerical variables
    # standardization
    x_s = quantify_data(data, True)

    # getting eigenvalues and eigenvectors
    # I commented it as it is not used in the following code
    # mean_vector = np.mean(x_s, axis=0)
    covariance_matrix = np.cov(x_s.T)
    #corelation_matrix= np.corrcoef(x_s.T)
    # extract eigenvalues
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    #eigen_values, eigen_vectors = np.linalg.eig(corelation_matrix)
    # ratio of explained variance (using the eigenvalues)

    sorted_eigenvalues = sorted(eigen_values, reverse=True)
    # I commented it as it is not used in the following code
    # tot_eig_vals = sum(eigen_values)
    # variance_explained = [(i / tot_eig_vals)*100 for i in sorted_eigenvalues ]

    # Sorting the eigenvectors accordingly to the eigenvalues
    eigen_vectors_values = [(np.abs(eigen_values[i]), eigen_vectors[:, i])
                            for i in range(len(eigen_vectors))]

    # Creating the top-2 eigenvectors matrix (4 x 2)
    top2_eigenvectors = np.hstack((eigen_vectors_values[0][1].reshape(2, -1),
                                   eigen_vectors_values[1][1].reshape(2, -1)))

    #plotting
    py.offline.init_notebook_mode(connected=True)

    # Get the PCA components (loadings)
    PCs = eigen_vectors

    # Use quiver to generate the basic plot
    fig = plt.figure(figsize=(5, 5))
    plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
               PCs[0, ], PCs[1, :],
               angles='xy', scale_units='xy', scale=1)

    # Add labels based on feature names (here just numbers)
    feature_names = np.array(["Alcohol","Malic_Acid","Ash","Ash_Alcanity",
                             "Magnesium","Total_Phenols","Flavanoids",
                             "Nonflavanoid_Phenols","Proanthocyanins",
                             "Color_Intensity","Hue","OD280","Proline",
                             "Customer_Segment"])
    for i, j, z in zip(PCs[1, :]+0.02, PCs[0, :]+0.02, feature_names):
        plt.text(j, i, z, ha='center', va='center')

    # Add unit circle
    circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
    plt.gca().add_artist(circle)

    # Ensure correct aspect ratio and axis limits
    plt.axis('equal')
    plt.xlim([-1.0, 1.0])
    plt.ylim([-1.0, 1.0])

    # Label axes
    plt.xlabel('PC 0')
    plt.ylabel('PC 1')

    # Done
    plt.show()
    
    
path = r'./data/wines_properties.csv'
data = prepare_and_load_data(path,skiprows=0)
a = pca_top2_extraction(data)
b = type(a)




#pca_exercise()


def pca_top2_extraction(data):
    x_s = quantify_data(data, True)  
    
    names = ["Alcohol","Malic_Acid","Ash","Ash_Alcanity",
                             "Magnesium","Total_Phenols","Flavanoids",
                             "Nonflavanoid_Phenols","Proanthocyanins",
                             "Color_Intensity","Hue","OD280","Proline",
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
    
    top2_withnames = pd.DataFrame(top2_eigenvectors, columns = [eigen_pairs[0][2],eigen_pairs[1][2]])
    
    return top2_withnames

pca_top2_extraction()
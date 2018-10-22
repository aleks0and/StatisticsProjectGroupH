import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA 

data = pd.read_csv(r"C:\Users\Michal\Documents\GitHub\StatisticsProjectGroupH\Project2\data\wines_properties.csv")

### dropping missing values 
data.dropna(how = "all", inplace=True)

# Storing only the numerical variables 
X = data.iloc[:, 0:-1]

#preimplemented PCA
my_pca = PCA(n_components=13)
new_projected_data = my_pca.fit(data)
PCs1 = new_projected_data.components_

#standardization
from sklearn.preprocessing import StandardScaler
X_s = StandardScaler().fit_transform(X)

#getting eigenvalues and eigenvectors
mean_vector = np.mean(X_s, axis = 0)
covariance_matrix = np.cov(X_s.T)

# extract eigen...
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

# ratio of explained variance (using the eigenvalues)
tot_eig_vals = sum(eigen_values)
sorted_eigenvalues = sorted(eigen_values, reverse=True)
variance_explained = [ (i / tot_eig_vals)*100 for i in sorted_eigenvalues ]

# Sorting the eigenvectors accordingly to the eigenvalues 
eigen_vectors_values = [ ( np.abs(eigen_values[i]), eigen_vectors[:, i] ) 
                        for i in range(len(eigen_values)) ]

## Creating the top-2 eigenvectors matrix (4 x 2)
top2_eigenvectors = np.vstack( ( eigen_vectors_values[0][1].reshape(1, -1), 
                             eigen_vectors_values[1][1].reshape(1, -1) ) )


#plotting
import matplotlib.pyplot as plt
import plotly as py
py.offline.init_notebook_mode(connected=True)

# Get the PCA components (loadings)
PCs = eigen_vectors

# Use quiver to generate the basic plot
fig = plt.figure(figsize=(5,5))
plt.quiver(np.zeros(PCs.shape[1]), np.zeros(PCs.shape[1]),
           PCs[0,:], PCs[1,:], 
           angles='xy', scale_units='xy', scale=1)

# Add labels based on feature names (here just numbers)
feature_names = np.array(["Alcohol","Malic_Acid","Ash","Ash_Alcanity",
                         "Magnesium","Total_Phenols","Flavanoids",
                         "Nonflavanoid_Phenols","Proanthocyanins",
                         "Color_Intensity","Hue","OD280","Proline",
                         "Customer_Segment"])
for i,j,z in zip(PCs[1,:]+0.02, PCs[0,:]+0.02, feature_names):
    plt.text(j, i, z, ha='center', va='center')

# Add unit circle
circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
plt.gca().add_artist(circle)

# Ensure correct aspect ratio and axis limits
plt.axis('equal')
plt.xlim([-1.0,1.0])
plt.ylim([-1.0,1.0])

# Label axes
plt.xlabel('PC 0')
plt.ylabel('PC 1')

# Done
plt.show()



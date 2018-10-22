from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


def plot_clusters(data, predicted_clusters, initialized_kmeans, number_of_clusters):
    for i in range(0, number_of_clusters):
        color = cm.nipy_spectral(float(i) / number_of_clusters)
        plt.scatter(data[predicted_clusters == i, 0],
                    data[predicted_clusters == i, 1],
                    s=50, c=color,
                    marker='o', edgecolor=color,
                    label='cluster %d' % (i+1))
    color = cm.nipy_spectral(float(number_of_clusters) / number_of_clusters)
    plt.scatter(initialized_kmeans.cluster_centers_[:, 0],
                initialized_kmeans.cluster_centers_[:,  1],
                s=250, marker='*',
                c=color, edgecolor='black',
                label='centroids')
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.tight_layout()
    plt.show()

# loading data and omiting specified number of rows, also dropping the rows with missing values.
def prepare_and_load_data(path, skip_rows):
    data = pd.read_csv(path, skiprows=skip_rows)
    data.dropna(how="all", inplace=True)
    data = data.iloc[:,0:-1]
    return data


def quantify_data(data, standardization):
    result = data.values
    if standardization:
        result = StandardScaler().fit_transform(result)
    return result


# REFERENCE POINT FROM CLASS EXERCISES
# def classes_exercise():
#     np.random.seed(1)
#     variables = ['Age', 'GPA', 'Income', 'Height']
#     labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4', 'ID_5']
#     X = np.random.random_sample([6,4])*10
#     df = pd.DataFrame(X, columns=variables,index=labels)
#     row_dist = pd.DataFrame(squareform(pdist(df,metric='euclidean')),
#                             columns = labels,
#                             index = labels)
#     row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
#
#     row_clusters_labeled = pd.DataFrame(row_clusters,
#                  columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
#                  index=['cluster %d' % (i + 1) for i in range(row_clusters.shape[0])])
#     row_dendogram = dendrogram(row_clusters, labels = labels)
#
#     print(df)
#     print(row_dist)
#     print("row clusters")
#     print(row_clusters)
#     print("row clusters labeled")
#     print(row_clusters_labeled)
#     plt.tight_layout()
#     plt.ylabel("Distance ")
#     plt.show()
#
#     return None
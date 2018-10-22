#implementation of 2nd project for Statistics
from hierarchical_cluster_exercise import hierarchical_cluster_analysis, assignment2_point2_specified
from PCA_exercise import pca_exercise
from PCA_components_comparison_exercise import principal_components_comparison_given_data, principal_components_comparison_given_data_all_in_one, principal_components_comparison_3by3
from Kmeans_extraction_exercise import best_k_for_kmeans, best_k_for_kmeans_given_data
from Kmeans_exercise import assignment2_point3, assignment2_point3_top2_eigenvalues, silhouette, original_vars_PCA

def main():

    #point 1: PCA
    pca_exercise()
    
    #point 2: hierarchical cluster analysis
    hierarchical_cluster_analysis()
    assignment2_point2_specified()
    
    #point 3: k-means cluster analysis
    assignment2_point3()
    assignment2_point3_top2_eigenvalues()
    
    #point 3.1: sillhouette
    silhouette()
    
    #point 3.2: Plot on the space of the first two dimensions of the PCA the clusters obtained with K-means
    assignment2_point3_top2_eigenvalues
    
    principal_components_comparison_given_data()
    principal_components_comparison_given_data_all_in_one()
    
    #point 3.3 and 3.4:
    original_vars_PCA()
    
    
    #point 4:
    
    #point 5:
    principal_components_comparison_given_data()
    principal_components_comparison_given_data_all_in_one()
    principal_components_comparison_3by3()
    
    
    
   
    
    
    
    return None
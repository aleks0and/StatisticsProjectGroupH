#implementation of 2nd project for Statistics
from hierarchical_cluster_exercise import hierarchical_cluster_analysis, assignment2_point2_specified
from PCA_exercise import pca_exercise
from PCA_components_comparison_exercise import principal_components_comparison_given_data, principal_components_comparison_given_data_all_in_one
from Kmeans_extraction_exercise import best_k_for_kmeans, best_k_for_kmeans_given_data
from Kmeans_exercise import assignment2_point3, assignment2_point3_top2_eigenvalues, silhouette, original_vars_PCA

def main():

    hierarchical_cluster_analysis()
    assignment2_point2_specified()
    pca_exercise()
    principal_components_comparison_given_data()
    principal_components_comparison_given_data_all_in_one()
    best_k_for_kmeans()
    best_k_for_kmeans_given_data()
    silhouette()
    assignment2_point3()
    assignment2_point3_top2_eigenvalues()
    original_vars_PCA()
    
    
    
    return None

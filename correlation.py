import pandas as pd
import numpy as np
from typing import List, Literal
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, cophenet
from utils import to_distance


def calculate_rolling_correlation(
        df: pd.DataFrame, 
        window_size: int,
        sliding_step: int | None = 1,
        methods: Literal['pearson', 'kendall', 'spearman'] | list | None = ['pearson', 'kendall', 'spearman']
) -> List[List[pd.DataFrame]]:
    """
    Calculate corrlation matrix between indices within each window for each time stamp.

    Args:
        df (DataFrame): dataframe with columns are the features of the index and with / without 'DATE' column.
        window_size (int): size of the sliding window.
        sliding_step (int): sliding step. Default to `1`.
        methods (str | list): methods to calculate correlation matirces. Can be a int or list. Methods can only be `'pearson'`, `'kendall'` or 'spearman'. \
            Default to `['pearson', 'kendall', 'spearman']`

    Returns:
        rolling_corrs: list with length of `len(df) - window_size + 1`, where each element list has length of `len(methods)` and contains different types of correlation matrices dertermined by `methods`.
    """

    if isinstance(methods, int):
        methods = [methods]

    df = df.drop(columns=['DATE'], errors='ignore')

    rolling_corrs = [
        [window_df.corr(method=method) for method in methods]
        for window_df in df.rolling(window=window_size, step=sliding_step)
    ][int(np.ceil((window_size-1)/sliding_step)):]
        
    return rolling_corrs


def generate_cophenetic_similarity(corr_matrices: List[List[pd.DataFrame]], filename: str | None = None):
    num_matrices = len(corr_matrices)
    cophenetic_correlation_similarity = np.zeros((num_matrices, num_matrices))
    condensed_distances = [squareform(np.mean([to_distance(corr_matrix) for corr_matrix in corr_matrices_three], axis=0)) for corr_matrices_three in corr_matrices]
    Z_matrices = [linkage(condensed_distance, method='average') for condensed_distance in condensed_distances]

    for i in range(num_matrices):
        for j in range(i, num_matrices):
            if i == j:
                cophenetic_correlation_similarity[i, j] = 1
            else:
                c_i, _ = cophenet(Z_matrices[i], condensed_distances[j])
                c_j, _ = cophenet(Z_matrices[j], condensed_distances[i])

                cophenetic_correlation_similarity[i, j] = cophenetic_correlation_similarity[j, i] = (c_i + c_j) / 2

    if filename is not None:
        np.save(filename, cophenetic_correlation_similarity)

    return cophenetic_correlation_similarity

def generate_meta_similarity(correlation_matrices: List[List[pd.DataFrame]], filename: str | None = None):
    flattened_matrices = [np.concatenate([corr_matrix.to_numpy().flatten() for corr_matrix in corr_matrices_three]) for corr_matrices_three in correlation_matrices]
    flattened_matrices = np.array(flattened_matrices)
    meta_similarity = np.corrcoef(flattened_matrices)

    if filename is not None:
        np.save(filename, meta_similarity)
    
    return meta_similarity


# def apply_pca(similarity:np.ndarray, variance_threshold=0.90):

#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(similarity)

#     pca = PCA(n_components=variance_threshold,svd_solver='full')
#     pca_transformed = pca.fit_transform(scaled_data)

#     return pca_transformed
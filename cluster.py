import inspect
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator
from utils import to_distance
from functools import wraps
from typing import Tuple, List


def add_dates(cluster_func):
    """
    Add arg `dates` with type DataFrame to the function. `dates` only has column (`'DATE'`,).
    And integrate the first output with type np.ndarray to DataFrame with columns (`'DATE'`, `'regime'`).
    """
    @wraps(cluster_func)
    def wrapper(*args, dates: pd.DataFrame | None = None, **kwargs):
        regimes, num_cluster = cluster_func(*args, **kwargs)
        if dates is not None:
            regimes_df = dates.copy()
            regimes_df['regime'] = regimes
            return regimes_df, num_cluster
        return regimes, num_cluster
    return wrapper


def df_column_to_numpy(arg_name: str, column_name: str):
    """
    Convert the `arg_name` into a DataFrame, and convert the column with name `column_name` to a numpy array.

    Args:
        arg_name (str): Arg of the dataframe to convert.
        column_name (str): Name of the column.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            if isinstance(bound_args.arguments[arg_name], pd.DataFrame):
                bound_args.arguments[arg_name] = bound_args.arguments[arg_name][column_name].to_numpy()

            # Call the original function with modified arguments
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def cluster_similarity(
    similarity: np.ndarray, 
    dates: pd.DataFrame,
    num_cluster: int, 
    find_optimal: bool | None = False
) -> Tuple[pd.DataFrame, int]:
    """
    Apply clustering (hierarchical clustering) on directly on the similairty matrix.
    
    Args:
        similarity (ndarray): Similarity matrix with shape (n,n), where n is the number of samples.
        dates (DataFrame): DataFrame only with column (`'DATE'`,), with a length of n.
        num_cluster (int): Number of the clusters. If `find_optimal` is `True`, this is the max number when finding the optimal number using elbow method.
            Otherwise, this is the number of the final clusters.
        find_optimal (bool): Indicates if to find the optimal number of clusters. Default to `False`.
    
    Returns:
        regimes (DataFrame): DateFrame only with columns (`'DATE'`, `'regime'`).
        num_cluster (int): Number of clusters.
    """

    return _cluster_similarity(
        similarity=similarity,
        dates=dates,
        num_cluster=num_cluster,
        find_optimal=find_optimal
    )


@add_dates
def _cluster_similarity(
    similarity: np.ndarray, 
    num_cluster: int, 
    find_optimal: bool | None = False
) -> Tuple[np.ndarray, int]:
    """
    Apply clustering (hierarchical clustering) on directly on the similairty matrix.
    
    Args:
        similarity (ndarray): Similarity matrix with shape (n,n), where n is the number of samples
        num_cluster (int): Numner of the clusters. If `find_optimal` is `True`, this is the max number when finding the optimal number using elbow method. 
            Otherwise, this is the number of the final clusters.
        find_optimal (bool): Indicates if to find the optimal number of clusters.
    """

    distance = to_distance(similarity)
    distance_condensed = squareform(distance, checks = False)
    Z = linkage(distance_condensed, method='average')

    if find_optimal:
        # Find optimal number of clusters using the elbow method
        wcss = []
        for k in range(1, num_cluster+1):
            labels = fcluster(Z, k, criterion='maxclust')
            wcss.append(wcss_score(similarity, labels))

        kl = KneeLocator(range(1, num_cluster+1), wcss, curve='convex', direction='decreasing')
        num_cluster = kl.elbow

    labels = fcluster(Z, num_cluster, criterion='maxclust')
    labels = labels - np.min(labels)

    return labels, num_cluster


def cluster(
    features: np.ndarray, 
    dates: pd.DataFrame,
    num_cluster: int, 
    find_optimal: bool | None = False
) -> Tuple[pd.DataFrame, int]:
    """
    K-means++ Cluster according to the features.
    
    Args:
        features (ndarry): Features with shape (n,m). where n is the number of examples and m is the dimenstion of the feature space.
        dates (DataFrame): DataFrame only with column (`'DATE'`,), with s length of n.
        num_cluster (int): Numner of the clusters. If `find_optimal` is `True`, this is the max number when finding the optimal number using elbow method.
            Otherwise, this is the number of the final clusters.
        find_optimal (bool): Indicates if to find the optimal number of clusters.

    Returns:
        regimes (DataFrame): Dataframe only with columns (`'DATE'`, `'regime'`).
        num_cluster (int): Number of clusters.
    """

    return _cluster(
        features=features,
        num_cluster=num_cluster,
        find_optimal=find_optimal,
        dates=dates
    )


@add_dates
def _cluster(
    features: np.ndarray, 
    num_cluster: int, 
    find_optimal: bool | None = False,
) -> Tuple[np.ndarray, int]:
    """
    Cluster according to the features.
    
    Args:
        features (ndarry): Features with shape (n,m). where n is the number of examples and m is the dimenstion of the feature space.
        num_cluster (int): Numner of the clusters. If `find_optimal` is `True`, this is the max number when finding the optimal number using elbow method.
            Otherwise, this is the number of the final clusters.
        find_optimal (bool): Indicates if find the optimal number of clusters.

    Returns:
        kmeans.labels_ (ndarray): A List of labels, i.e., the regimes.
        num_cluster (int): Number of cluster, i.e., the total number of different regimes.
    """

    if find_optimal:
        # Find optimal number of clusters using the elbow method
        wcss = []
        for k in range(1, num_cluster+1):
            kmeans = KMeans(n_clusters=k, init='k-means++')
            kmeans.fit(features)
            wcss.append(kmeans.inertia_)

        kl = KneeLocator(range(1, num_cluster+1), wcss, curve='convex', direction='decreasing')
        num_cluster = kl.elbow

    kmeans = KMeans(n_clusters=num_cluster)
    kmeans.fit(features)

    return kmeans.labels_, num_cluster


def wcss_score(features: np.ndarray, labels: np.ndarray, normalize: bool | None = False) -> float:
    """
    Calculate wcss score.
    
    Args:
        features (ndarray): Features with shape (n,m), where n is the number of examples and m is the dimension of the feature space.
        labels (ndarray): Cluster labels with shape (n,).

    Returns:
        wcss (float): wcss score.
    """
    wcss = 0.0
    for label in np.unique(labels):
        points_cluster = features[labels==label]
        centroid = np.mean(points_cluster, axis=0)
        squared_distance = np.sum((points_cluster - centroid)**2)
        wcss += squared_distance

    if normalize:
        tss = np.sum((features-np.mean(features, axis=0))**2)
        wcss = wcss / tss

    return wcss


def assess_clustering_results(features: np.ndarray, regimes: pd.DataFrame):
    """
    Assess clustering results using 'wcss', 'wcss_norm', 'silhouette', 'davies_bouldin', 'calinski_harabasz'.
    
    Args:
        features (ndarray): Features with shape (n,m). where n is the number of examples and m is the dimension of the feature space.
        regimes (DataFrame): DataFrame with the column (`'regime'`, ).

    Returns:
        metrics (Series): Series with metrics as indices and scores as values.
    """

    metrics = _assess_clustering_results(features=features, labels=regimes)

    return metrics


@df_column_to_numpy(arg_name='labels', column_name='regime')
def _assess_clustering_results(features: np.ndarray, labels: np.ndarray) -> pd.Series:
    """
    Assess clustering results using 'wcss', 'wcss_norm', 'silhouette', 'davies_bouldin', 'calinski_harabasz'.
    
    Args:
        features (ndarray): Features with shape (n,m). where n is the number of examples and m is the dimension of the feature space.
        labels (ndarray): Cluster labels with shape (n,).

    Returns:
        metrics (Series): Series with metrics as indices and scores as values.
    """

    metrics = {
        'wcss': wcss_score(features, labels),
        'wcss_norm': wcss_score(features, labels, True),
        'silhouette': silhouette_score(features, labels),
        'davies_bouldin': davies_bouldin_score(features, labels),
        'calinski_harabasz': calinski_harabasz_score(features, labels)
    }

    return pd.Series(metrics)


def assess_clustering_on_returns(dfs: List[pd.DataFrame], regimes: pd.DataFrame):
    """
    Analyse clustering obtained from features by treating the returns as 'features'.

    Args:
        dfs (list): List of Dataframes, where each dataframe contains columns (`'DATE'`, ...) where other columns are trailing or forward returns.
        regimes (DataFrame): DataFrame that contains columns (`'DATE'`, `'regime'`).

    Returns:
        out (DataFrame): DataFrame with columns (`'index'`, `'metric'`, ...) where other columns are trailing or forward returns.
    """

    regimes = regimes[['DATE', 'regime']]
    dfs_with_regime = []
    for df in dfs:
        df_with_regime = pd.merge(df, regimes, on='DATE').dropna()
        df_with_regime.Name = df.Name
        dfs_with_regime.append(df_with_regime)

    results = {
        df.Name : pd.DataFrame({
            column_name : assess_clustering_results(np.expand_dims(returns_series.to_numpy(),-1), df)
            for column_name, returns_series in df.drop(columns=['DATE', 'regime']).items()
        })
        for df in dfs_with_regime
    }

    return pd.concat(results, axis=0, names=['index', 'metric']).swaplevel(0,1).reset_index().sort_values(by=['metric', 'index'], ascending=False)

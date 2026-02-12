import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Dict, Any
from config import PCA_VARIANCE_THRESHOLD

# cluster stocks based on PCA of the correlation matrix of returns.
# dynamically determines n_components based on explained variance ratio.
def cluster_assets(prices: pd.DataFrame, n_clusters: int = 6) -> Dict[Any, List[str]]:

    # calc daily returns and drop any nan values
    returns: pd.DataFrame = prices.pct_change().dropna()
    # calculate the correlation matrix of returns.
    # this matrix will be input for PCA.
    # fillna(0) is used to handle cases where correlation might be nan for non-liquid stocks or short periods.
    corr: pd.DataFrame = returns.corr().fillna(0)

    # PCA
    # init PCA with all components allowing us to calc the explained variance ratio
    # for each component and determine optimal number of components dynamically.
    pca = PCA(n_components=None)

    # fit PCA to the correlation matrix and transform it to get PCs.
    pcs_full: np.ndarray = pca.fit_transform(corr)

    # determine optimal number of components based on explained variance ratio
    cumulative_variance_ratio: np.ndarray = np.cumsum(pca.explained_variance_ratio_)

    # find smallest number of components whose cumulative explained variance
    # meets or exceeds the PCA_VARIANCE_THRESHOLD
    n_components_optimal: int = np.where(cumulative_variance_ratio >= PCA_VARIANCE_THRESHOLD)[0]

    if n_components_optimal.size > 0:
        # add 1 because np.where returns 0 based indices, and we want the count of components
        n_components_optimal = int(n_components_optimal[0]) + 1
    else:
        # if the threshold is never met, use all available components.
        n_components_optimal = corr.shape[0]

    # ensure n_components_optimal does not exceed the number of tickers
    n_components_optimal = min(n_components_optimal, corr.shape[0])

    # use only the optimal number of PCs for clustering
    pcs: np.ndarray = pcs_full[:, :n_components_optimal]

    # KMeans clustering
    # group assets based on their similarity in PCA space.

    # n_clusters must not exceed the number of samples available after PCA
    effective_n_clusters: int = min(n_clusters, pcs.shape[0])
    # edge case where there is no data or too few samples for clustering.
    if effective_n_clusters == 0:
        return {}

    # init KMeans
    # n_init=10 to run algorithm 10 times with different centroid seeds and choose the best result.
    kmeans = KMeans(n_clusters=effective_n_clusters, random_state=42, n_init=10)
    # Fit KMeans to the PCs and predict cluster label for each asset.
    labels: np.ndarray = kmeans.fit_predict(pcs)

    # map cluster labels back to original ticker symbols
    clusters: Dict[Any, List[str]] = {}
    for label, ticker in zip(labels, corr.columns):
        clusters.setdefault(label, []).append(ticker)

    return clusters

# performs PCA on the correlation matrix of returns and returns the first two components for visualization.
def get_pca_for_visualization(prices: pd.DataFrame) -> (pd.DataFrame, list):

    returns = prices.pct_change().dropna()
    corr = returns.corr().fillna(0)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(corr)

    pca_df = pd.DataFrame(pcs, columns=['PC1', 'PC2'], index=corr.index)
    return pca_df


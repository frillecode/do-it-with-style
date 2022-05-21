import os
import time
import argparse

import pandas as pd
import numpy as np
import ndjson

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


def grid_search_clustering(X, parameters, algorithm, use_silhouette=True, res_path=None):
    '''Given parameter space, fit many kmeans models 
    and track performance.
    Parameters
    ----------
    X : np.array
        Training data.
    parameters : dict
        Named parameters, where key is parameter name
        and value a list of values to try.
    algorithm : instance
        tag of algorithm to use for the clustering.
        Options are kmeans & agglomerative are implemented. 
    use_silhouette : bool
        Calculate silhouette score for each model?
        If false, only inertia will be included in output.
    res_path : str, optional
        If specified, results will be dumped there, by default None
    Returns 
    -------
    list
        if res_path not specified, returns log of results
    '''

    param_grid = ParameterGrid(parameters)

    # pick cluster algorithm
    grid_search_results = []
    for params in param_grid:
        t0 = time.time()
        km = algorithm(**params)
        km.fit(X)
        t_bench = time.time() - t0

        res = {}

        if use_silhouette:
            silhouette_ = silhouette_score(
                X,
                km.labels_,
                metric='cosine'
            )
            res['silhouette'] = silhouette_

        res['train_time'] = t_bench
        res['params'] = params

        grid_search_results.append(res)

    if res_path:
        with open(res_path, 'w') as fout:
            ndjson.dump(grid_search_results, fout)

    return grid_search_results
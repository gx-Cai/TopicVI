import scanpy as sc
import warnings
import numpy as np
from sklearn.metrics import davies_bouldin_score, normalized_mutual_info_score
import pandas as pd

def cluster_optimal_resolution(
    adata,
    label_key,
    cluster_key,
    cluster_function=None,
    metric=None,
    resolutions=None,
    use_rep=None,
    force=False,
    verbose=True,
    return_all=False,
    metric_kwargs=None,
    **kwargs,
):
    """
    [borrow from scib package]
    
    Optimised clustering 

    Leiden, louvain or any custom clustering algorithm with resolution optimised against a metric

    :param adata: anndata object
    :param label_key: name of column in adata.obs containing biological labels to be
        optimised against
    :param cluster_key: name and prefix of columns to be added to adata.obs during clustering.
        Each resolution will be saved under "{cluster_key}_{resolution}", while the optimal clustering will be under ``cluster_key``.
        If ``force=True`` and one of the keys already exists, it will be overwritten.
    :param cluster_function: a clustering function that takes an anndata.Anndata object. Default: Leiden clustering
    :param metric: function that computes the cost to be optimised over. Must take as
        arguments ``(adata, label_key, cluster_key, **metric_kwargs)`` and returns a number for maximising
        Default is :func:`~scib.metrics.nmi()`
    :param resolutions: list of resolutions to be optimised over. If ``resolutions=None``,
        by default 10 equally spaced resolutions ranging between 0 and 2 will be used (see :func:`~scib.metrics.get_resolutions`)
    :param use_rep: key of embedding to use only if ``adata.uns['neighbors']`` is not
        defined, otherwise will be ignored
    :param force: whether to overwrite the cluster assignments in the ``.obs[cluster_key]``
    :param verbose: whether to print out intermediate results
    :param return_all: whether to results for all resolutions
    :param metric_kwargs: arguments to be passed to metric
    :param kwargs: arguments to pass to clustering function
    :returns:
        Only if ``return_all=True``, return tuple of ``(res_max, score_max, score_all)``
        ``res_max``: resolution of maximum score;
        ``score_max``: maximum score;
        ``score_all``: ``pd.DataFrame`` containing all scores at resolutions. Can be used to plot the score profile.

    If you specify an embedding that was not used for the kNN graph (i.e. ``adata.uns["neighbors"]["params"]["use_rep"]`` is not the same as ``use_rep``),
    the neighbors will be recomputed in-place.
    """

    def call_cluster_function(adata, res, resolution_key, cluster_function, **kwargs):
        if resolution_key in adata.obs.columns:
            warnings.warn(
                f"Overwriting existing key {resolution_key} in adata.obs", stacklevel=2
            )

        # check or recompute neighbours
        knn_rep = adata.uns.get("neighbors", {}).get("params", {}).get("use_rep")
        if use_rep is not None and use_rep != knn_rep:
            print(f"Recompute neighbors on rep {use_rep} instead of {knn_rep}")
            sc.pp.neighbors(adata, use_rep=use_rep)

        # call clustering function
        print(f"Cluster for {resolution_key} with {cluster_function.__name__}")
        cluster_function(adata, resolution=res, key_added=resolution_key, **kwargs)

    if cluster_function is None:
        cluster_function = sc.tl.leiden

    if cluster_key is None:
        cluster_key = cluster_function.__name__

    if metric is None:
        metric = nmi

    if metric_kwargs is None:
        metric_kwargs = {}

    if resolutions is None:
        resolutions = np.linspace(0.1, 1.0, 10)

    score_max = 0
    res_max = resolutions[0]
    clustering = None
    score_all = []

    for res in resolutions:
        resolution_key = f"{cluster_key}_{res}"

        # check if clustering exists
        if resolution_key not in adata.obs.columns or force:
            call_cluster_function(
                adata, res, resolution_key, cluster_function, **kwargs
            )

        # score cluster resolution
        score = metric(adata, label_key, resolution_key, **metric_kwargs)
        score_all.append(score)

        if verbose:
            print(f"resolution: {res}, {metric.__name__}: {score}", flush=True)

        # optimise score
        if score_max < score:
            score_max = score
            res_max = res
            clustering = adata.obs[resolution_key]

    if verbose:
        print(f"optimised clustering against {label_key}")
        print(f"optimal cluster resolution: {res_max}")
        print(f"optimal score: {score_max}")

    score_all = pd.DataFrame(
        zip(resolutions, score_all), columns=["resolution", "score"]
    )

    # save optimal clustering in adata.obs
    if cluster_key in adata.obs.columns:
        warnings.warn(
            f"Overwriting existing key {cluster_key} in adata.obs", stacklevel=2
        )
    adata.obs[cluster_key] = clustering

    if return_all:
        return res_max, score_max, score_all
    return res_max, score_max

def nmi(adata, label_key, cluster_key, **kwargs):
    return normalized_mutual_info_score(
        adata.obs[label_key], 
        adata.obs[cluster_key], 
        **kwargs
    )

def inverse_davies_bouldin_score(
        adata, label_key, cluster_key, 
        use_rep='X_pca',
        **metric_kwargs
    ):
    if adata.obs[cluster_key].nunique() == 1:
        return 0
    return 1 / davies_bouldin_score(
        adata.obsm[use_rep], 
        adata.obs[cluster_key]
    )

def run_default_cluster(
        adata_, layer, batch_key, 
        max_cells = 10000,
        optimal_resolution = True,
        resolution = 1.0
    ):
    adata = adata_.copy()
    if layer is not None:
        adata.X = adata.layers[layer]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    try:
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000, batch_key=batch_key)
    except Exception as e:
        print("Error in highly_variable_genes 'seurat_v3', will try to run with default params:", e)
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=2000)
    
    sc.pp.pca(adata, use_highly_variable=True)
    if batch_key is not None and adata.obs[batch_key].nunique() > 1:
        # run harmony
        sc.external.pp.harmony_integrate(adata, key=batch_key)
        X_pca_key = 'X_pca_harmony'
    else:
        X_pca_key = 'X_pca'

    X_pca = adata.obsm[X_pca_key]
    if adata.shape[0] > max_cells:
        warnings.warn(f"The number of cells is larger than {max_cells}, will subsample to {max_cells} cells for speeding up", UserWarning)
        sc.pp.subsample(adata, n_obs=max_cells, )

    sc.pp.neighbors(adata, metric='euclidean', use_rep=X_pca_key) # 'cosine'
    if not optimal_resolution:  
        sc.tl.leiden(adata, resolution=resolution, key_added='default_cluster')
    else:
        cluster_optimal_resolution(
            adata, 
            label_key=None, 
            cluster_key='default_cluster', 
            use_rep=X_pca_key,
            cluster_function=sc.tl.leiden,
            metric=inverse_davies_bouldin_score,
            resolutions=np.linspace(0.1, 1.0, 10)
        )
         

    adata_.obs['default_cluster'] = adata.obs['default_cluster'].astype(object)
    adata_.obs['default_cluster'] = adata_.obs['default_cluster'].fillna('Unknown')
    adata_.obsm['X_pca'] = X_pca
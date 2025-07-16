import scanpy as sc
import numpy as np
from scipy.stats import median_abs_deviation
from scipy.sparse import issparse, csr_matrix
from .utils import verbose_print, is_count_data

def filtering_reads(adata, verbose = True, store_qc = False):

    def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            np.median(M) + nmads * median_abs_deviation(M) < M
        )
        return outlier
    adata = adata.copy()
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    raw_varinfo = adata.var_keys()
    raw_obsinfo = adata.obs_keys()
    
    # mitochondrial genes
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes.
    adata.var["hb"] = adata.var_names.str.contains(("^HB[^(P)]"))

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )

    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", 5)
        | is_outlier(adata, "log1p_n_genes_by_counts", 5)
        | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
    )
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 3) | (
        adata.obs["pct_counts_mt"] > 8
    )
    
    raw_nobs = adata.n_obs
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    
    if verbose:
        print(f"Total number of cells: {raw_nobs}")
        print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")

    if store_qc:    
        qc_varkeys = list(set(adata.var_keys()) - set(raw_varinfo))
        qc_obskeys = list(set(adata.obs_keys()) - set(raw_obsinfo))
        
        adata.uns['qc'] = {
            'obs': adata.obs[qc_obskeys],
            'var': adata.var[qc_varkeys]
        }
    
    adata.obs = adata.obs[raw_obsinfo]
    adata.var = adata.var[raw_varinfo]
    
    return adata

def sizeFactors_sparse(counts):
    """
    median-geometric-mean implement.
    """
    logcnt = counts[~np.array(((counts !=0).sum(axis=1) ==0)).flatten(),:].copy()
    logcnt.data = np.log(counts.data)
    
    # calculate the geometric means along columns
    def loggeomeans_vector(i):
        return logcnt.getrow(i).data.mean()
    
    loggeomeans = np.array([loggeomeans_vector(i) for i in range(counts.shape[0])])
    # calculate the logratios
    logratios = logcnt - loggeomeans[:,None]
    return np.array(np.exp(np.median(logratios,axis=0))).flatten()

def preprocess_adata(
    adata, 
    run_qc_filtering = True,
    nhvg = 2000, 
    batch_key = None,
    verbose = True
):
    
    if not is_count_data(adata.X):
        raise ValueError("adata.X must be count data (non-negative integers). Please check your data.")

    if not issparse(adata.X):
        verbose_print("Converting adata.X to sparse matrix.", verbose=verbose)
        adata.X = csr_matrix(adata.X)
    
    verbose_print("Assigning size factors.", verbose=verbose)
    assign_size_factors(adata)

    if run_qc_filtering:
        verbose_print("Running QC filtering.", verbose=verbose)
        adata = filtering_reads(adata, verbose=verbose)

    sc.pp.filter_genes(adata, min_cells=1)
    
    if batch_key:
        batch_num = adata.obs[batch_key].value_counts()
        adata.obs[batch_key] = adata.obs[batch_key].astype('category')
        if (batch_num == 1).any():
            print("Some batch only contains 1 cell. Will not use batch to get hvg.")
            batch_key = None
    
    verbose_print("Running basic preprocessing steps.", verbose=verbose)
    preprocess_adata_basic(adata)

    if nhvg:
        verbose_print(f"Finding {nhvg} highly variable genes.", verbose=verbose)
        try:
            sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=nhvg, layer="counts", subset=True, batch_key=batch_key)
        except Exception as e:
            print("Error Occured in highly_variable_genes 'seurat_v3', to avoid the error, may pass `batch_key=None`")
            print("Error in highly_variable_genes 'seurat_v3', will try to run with default params:", e)
            sc.pp.highly_variable_genes(adata, n_top_genes=nhvg, subset=True, batch_key=batch_key)
    

    verbose_print("Finished. Resetting adata.X to counts layer.", verbose=verbose)
    adata.X = adata.layers["counts"]
    # del adata.uns['log1p']
    return adata

def preprocess_adata_basic(adata):
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers['normalized'] = adata.X.copy()
    sc.pp.pca(adata)

def assign_size_factors(adata):
    cell_total = adata.X.sum(axis=1)
    sf = np.array(cell_total / np.exp(np.log(cell_total).mean())).flatten()
    sf[np.isnan(sf)] = 1
    adata.obs['size_factor'] = sf


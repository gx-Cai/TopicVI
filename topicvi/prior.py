import scanpy as sc
import numpy as np
import pandas as pd
import os
import json

data_base_path = os.path.join(os.path.dirname(__file__), "data")


def summary_of_builtin_priors():
    print(
        """
        Summary of built-in priors:

        Cell Types Related:
        - CellMarkerDB. use `search_cellmarker_db` to search for cell markers.
        - ScTypeDB. use `search_sctype_db` to search for cell types.
        - Meta-Program. use `get_metaprogram` to get meta-programs.

        Common Used Annotations:
        (From Enrichr Database and MSigDB)
        use get_priors to get the gene sets, specify the source as below.
        - GO. (GOBP, GOMF, GOCC)
        - KEGG.
        - Reactome.
        - Hallmark.
        - Immune Related.
        - TF. 
        - Tissue. (Tissue specific gene sets from DAVID database)

        Others:
        - DEGenes. use `get_de_genes` to get differentially expressed genes.
        - Random. use `get_random_select_gs` to get random gene sets.
        """
    )


def add_prior_to_adata(
        adata, background_prior, cluster_prior, 
        background_min_genes=10, cluster_min_genes=5,
        key_added="annotation", 
        overwrite=False
):
    if key_added in adata.uns.keys() and not overwrite:
        raise ValueError(
            f"Key {key_added} already exists in adata.uns. Set `overwrite` to True to overwrite it."
        )

    adata.uns[key_added] = {
        "background": clean_prior_dict(background_prior, adata, min_genes=background_min_genes),
        "clusters": clean_prior_dict(cluster_prior, adata, min_genes=cluster_min_genes),
    }


### Utils for prior knowledge


def clean_prior_dict(prior: dict, adata, min_genes=5):
    def __change_prior_key(prior):
        # change '/' to '-' in the keys of the prior dict
        return {k.replace("/", "-"): v for k, v in prior.items()}

    prior = {k: [i for i in v if i in adata.var_names] for k, v in prior.items()}
    prior = {k: v for k, v in prior.items() if len(v) > min_genes}
    prior = __change_prior_key(prior)
    # print({k: len(v) for k, v in prior.items()})
    return prior


def filter_prior_dict(prior: dict, keywords: list):
    if len(keywords) == 0:
        return prior
    prior = {k: v for k, v in prior.items() if any([kw in k for kw in keywords])}
    return prior


### Get prior knowledge


def get_de_genes(adata, n=50, cell_type_key="cell_type", layer=None):
    if layer is not None:
        nomalized = adata.copy()
        nomalized.X = nomalized.layers[layer]
    else:
        from . import is_count_data
        if is_count_data(adata):
            nomalized = adata.copy()
            sc.pp.normalize_total(nomalized, target_sum=1e4)
            sc.pp.log1p(nomalized)
        else:
            nomalized = adata.copy()
    assert (
        cell_type_key in nomalized.obs.columns
    ), f"Cell type key `{cell_type_key}` not found in adata.obs."

    sc.tl.rank_genes_groups(
        nomalized, groupby=cell_type_key, method="wilcoxon", use_raw=False
    )
    ground_genesets = {
        ct: sc.get.rank_genes_groups_df(
            nomalized, group=ct, pval_cutoff=0.05, log2fc_min=1
        )
        .sort_values("pvals_adj")
        .head(n)["names"]
        .tolist()
        for ct in nomalized.obs[cell_type_key].cat.categories
    }
    return ground_genesets


# search for genesets based on Cell_marker
def search_cellmarker_db(
    cell_types, tissue_type=None, cancer_type="Normal", search="coarse"
):
    path = os.path.join(data_base_path, "CellMarkerDB.Human.xlsx")
    if isinstance(cell_types, str):
        cell_types = [cell_types]
    print(
        "Searching for cell markers",
        "with tissue type",
        tissue_type,
        "and cancer type",
        cancer_type,
    )
    ret = {}
    df = pd.read_excel(path)

    if tissue_type is not None:
        query_str = "tissue_type == @tissue_type and cancer_type == @cancer_type"
    else:
        query_str = "cancer_type == @cancer_type"

    df = df.query(query_str).copy()
    if df.shape[0] == 0:
        raise ValueError(
            "No cell markers found with the given tissue type and cancer type. PLEASE CHECK THE INPUT."
        )
    # to avoid warnning
    df.loc[:, "cell_name"] = df["cell_name"].str.lower()

    all_cts = df["cell_name"].unique()
    for ct in cell_types:
        coarse_flag = False
        ctl = ct.lower()
        if search == "strict":
            markers = df.query("cell_name == @ctl")["Symbol"].tolist()
            markers = pd.Series(markers, dtype="object").dropna().unique()
            ret[ct] = markers.tolist()
            if len(markers) == 0:
                print(
                    f"Cell type {ct} not found in the database. Will try coarse search."
                )
                coarse_flag = True
        if coarse_flag or search == "coarse":
            selected_cts = [i for i in all_cts if ctl in i]  # noqa: F841
            markers = df.query("cell_name in @selected_cts")["Symbol"].tolist()
            markers = pd.Series(markers).dropna().unique()
            ret[ct] = markers.tolist()
    return ret


def load_sctype_db():
    sctype_db = pd.read_excel(os.path.join(data_base_path, "ScTypeDB.full.xlsx"))
    sctype_db.rename(
        columns={
            "cellName": "cell_type",
            "tissueType": "tissue_type",
            "geneSymbolmore1": "positive_markers",
            "geneSymbolmore2": "negative_markers",
            "shortName": "cell_type_abbv",
        },
        inplace=True,
    )
    return sctype_db


def search_sctype_db(cell_type, tissue_type=None, search="coarse", with_negative=False):
    if with_negative:
        raise NotImplementedError(
            "Negative markers not supported yet. Please set with_negative to False."
            "With `load_sctype_db` function, you can get the full database."
        )
    sctype_db = load_sctype_db()
    print("Loading cell type database")
    if tissue_type is not None:
        print("Searching for tissue type", tissue_type)
        query_str = "tissue_type == @tissue_type"
        sctype_db = sctype_db.query(query_str).copy()

    print("Searching for cell type", cell_type)
    assert search in ["strict", "coarse"], "Search method not supported."

    if search == "strict":
        ret = sctype_db.query("cell_type == @cell_type").copy()
        if ret.shape[0] == 0:
            print(
                f"Cell type {cell_type} not found in the database. Will try coarse search."
            )
            search = "coarse"
    if search == "coarse":
        ret = sctype_db.query(
            "cell_type.str.contains(@cell_type)", engine="python"
        ).copy()
        if ret.shape[0] == 0:
            print(
                f"Cell type {cell_type} not found in the database. PLEASE CHECK THE INPUT."
            )

    # return a dict with name and gene list
    ret["positive_markers"] = ret["positive_markers"].str.split(",").tolist()
    ret = ret[["cell_type", "positive_markers"]].to_dict(orient="records")
    ret = {i["cell_type"]: i["positive_markers"] for i in ret}
    return ret


def get_priors(
    source,
):
    assert source in [
        "GOBP",
        "GOMF",
        "GOCC",
        "KEGG",
        "Reactome",
        "Hallmark",
        "Immune",
        "TF",
        "Tissue"
    ], "Source not supported."
    if source == 'Tissue':
        source = 'DAVID.tissue_up'
    elif source in ["Immune", "TF"]:
        source = f"msigdb.{source}"
    else:
        source = f"enrichr.{source}"

    with open(os.path.join(data_base_path, f"{source}.json"), "r") as f:
        ret = json.load(f)
    return ret


def get_random_select_gs(n_select, source_gs=None, seed=0):
    np.random.seed(seed)
    if source_gs is None:
        reactome_gene_sets = get_priors("Reactome")
        hallmark_gene_sets = get_priors("Hallmark")
        source_gs = {**reactome_gene_sets, **hallmark_gene_sets}
    random_select = np.random.choice(range(len(source_gs)), n_select, replace=False)
    random_select = {
        list(source_gs.keys())[i]: list(source_gs.values())[i] for i in random_select
    }
    return random_select


def load_metaprograms(cell_type = 'all'):
    """load meta-programs generated by (Avishai Gavish, et al. 2023, Nature)
    Downloaded from https://www.weizmann.ac.il/sites/3CA/search-genes.

    Args:
        cell_type (str, optional): cell types. Defaults to 'all'.
        
        valid cell types are:
        - 'all'
        - 'Malignant', 'B cells', 'Endothelial', 'Epithelial', 'Fibroblasts', 'Macrophages', 'CD4 T cells', 'CD8 T cells'

    Returns:
        dict: meta-programs, keys are names of meta-programs, values are gene lists.
    """
    valid_cell_types = [
        'all', 
        'Malignant', 'B cells', 'Endothelial', 'Epithelial', 
        'Fibroblasts', 'Macrophages', 'CD4 T cells', 'CD8 T cells'
    ]
    assert cell_type in valid_cell_types, f"Cell type {cell_type} not supported. Please choose from {valid_cell_types}."
    file = os.path.join(data_base_path, "meta_programs.xlsx")
    if cell_type != 'all':
        metaprogram = pd.read_excel(file, sheet_name=cell_type)
        return {name.replace('/','-'): metaprogram[name].tolist() for name in metaprogram.columns}
    else:
        ret = {}
        for ct in valid_cell_types[1:]:
            metaprogram = pd.read_excel(file, sheet_name=ct)
            ret[ct] = {name.replace('/','-'): metaprogram[name].tolist() for name in metaprogram.columns}
        return ret

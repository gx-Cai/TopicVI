## Author: CGX
## Time: 2024 10 21
## Description: Metrics for evaluating the quality of topics.
# Modify from: https://github.com/JinmiaoChenLab/scTM/ and https://github.com/MIND-Lab/OCTIS

import itertools
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import issparse
import math
from .utils import is_count_data, array_nlargest

def rbo_min(list1, list2, p, depth=None):
    """Tight lower bound on Rank Biased Overlap (RBO) implementation."""

    def set_at_depth(lst, depth):
        ans = set()
        for v in lst[:depth]:
            if isinstance(v, set):
                ans.update(v)
            else:
                ans.add(v)
        return ans

    def overlap(list1, list2, depth):
        set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
        len_intersection, len_set1, len_set2 = len(set1.intersection(set2)), len(set1), len(set2)
        aggrement = 2 * len_intersection / (len_set1 + len_set2)
        return aggrement * min(depth, len(list1), len(list2))
        # NOTE: comment the preceding and uncomment the following line if you want
        # to stick to the algorithm as defined by the paper
        # return len_intersection

    depth = min(len(list1), len(list2)) if depth is None else depth
    x_k = overlap(list1, list2, depth)
    log_term = x_k * math.log(1 - p)
    sum_term = sum(
        p**d / d * (overlap(list1, list2, d) - x_k) for d in range(1, depth + 1)
    )
    return (1 - p) / p * (sum_term - log_term)

def densify_data(adata, layer):
    if layer is None:
        if issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X
    else:
        if issparse(adata.layers[layer]):
            data = adata.layers[layer].toarray()
        else:
            data = adata.layers[layer]
    if is_count_data(data):
        # if is count data, normalized and log1p the data for better performance
        # sc.pp.normalize_total()
        data = data / data.sum(axis=1, keepdims=True) * 1e4
        data = np.log1p(data)

    return data.astype("float32")

class TopicMetrics:
    def __init__(
        self, 
        adata, 
        layer = None,
        topic_comp = None, # [n_topics x n_genes]
        topic_comp_key = 'topic_by_gene',
        topic_prop=None,
        topic_prop_key = 'topic_by_sample', 
        topk=20, 
        topic_comp_min=0,
        rbo_p=0.8,
        coherence_quantile=0.75,
        coherence_vmax=1,
        coherence_norm=False
    ):
        self.data = densify_data(adata, layer)
        if topic_prop is None:
            self.topic_prop = adata.obsm[topic_prop_key] # n_obs x n_topics
        else:
            self.topic_prop = topic_prop
        self.topk = topk
        if topic_comp is None:
            self.beta = adata.varm[topic_comp_key].T
        else:
            self.beta = topic_comp # n_topics x n_genes
        self.gene_names = adata.var_names
        self.obs_names = adata.obs_names
        self.topics = array_nlargest(self.beta, self.topk, vmin=topic_comp_min) # n_topics x topk

        # input check
        assert self.beta.shape[1] == len(self.gene_names)
        assert self.topic_prop.shape[0] == self.data.shape[0] == len(self.obs_names)

        self.rbo_p = rbo_p
        self.coherence_quantile = coherence_quantile
        self.coherence_vmax = coherence_vmax
        self.coherence_norm = coherence_norm

    def get_topics(self):
        return [self.gene_names[i] for i in self.topics]

    def get_topic_sparseness(self, topic_prop=None):
        """
        The sparseness of array x is a real number in [0, 1], where sparser array
        has value closer to 1. Sparseness is 1 if the vector contains a single
        nonzero component and is equal to 0 if all components of the vector are
        the same

        modified from Hoyer 2004: [sqrt(n)-L1/L2]/[sqrt(n)-1]

        adapted from nimfa package: https://nimfa.biolab.si/
        """
        from math import sqrt  # faster than numpy sqrt

        if topic_prop is None:
            topic_prop = self.topic_prop

        x = topic_prop
        eps = np.finfo(x.dtype).eps if "int" not in str(x.dtype) else 1e-9
        x = x / x.sum(axis=1, keepdims=True)
        n = x.size

        # measure is meant for nmf: things get weird for negative values
        if np.min(x) < 0:
            x -= np.min(x)

        # patch for array of zeros
        if np.allclose(x, np.zeros(x.shape), atol=1e-6):
            return 0.0

        L1 = abs(x).sum()
        L2 = sqrt(np.multiply(x, x).sum())
        sparseness_num = sqrt(n) - (L1 + eps) / (L2 + eps)
        sparseness_den = sqrt(n) - 1

        return sparseness_num / sparseness_den

    def get_rbo(self, p=None, individual=False):
        p = self.rbo_p if p is None else p
        topics = self.topics
        num_topics = len(topics)
        rbos = []
        for i in range(num_topics):
            rbos_topics = []
            for j in range(num_topics):
                if i != j:
                    rbos_topics.append(1 - rbo_min(topics[i], topics[j], p=p))
                    # rbos_topics.append(1 - rbo.RankingSimilarity(topics[i], topics[j]).rbo(p = p))
            rbos.append(np.min(rbos_topics))
        if individual:
            return rbos
        else:
            return np.mean(rbos)

    def get_coherence_NPMI(
        self, quantile=None, individual=False, 
    ):
        def get_cell_probability(data, wi, quantiles, wj=None, vmax=self.coherence_vmax):
            ti = np.max((quantiles[wi], vmax))
            F_wi = (data[:, wi] >= ti)
            if wj is None: return F_wi.mean()

            # Find probability that they are not both zero
            tj = np.max((quantiles[wj], vmax))
            F_wj = (data[:, wj] >= tj)
            D_wj = F_wj.mean()
            D_wi_wj = (F_wi & F_wj).mean()

            return D_wj, D_wi_wj

        data = self.data
        quantile = self.coherence_quantile if quantile is None else quantile
        if quantile > 0:
            quantiles = np.quantile(data, q=quantile, axis=0)
        else:
            quantiles = np.zeros(data.shape[1])

        TC = []
        topics = self.topics
        for beta_topk in topics:
            TC_k = np.zeros(shape=(len(beta_topk), len(beta_topk)))

            for i, gene in enumerate(beta_topk):
                # get D(w_i)
                D_wi = get_cell_probability(data, gene, quantiles)
                for j in range(i+1, len(beta_topk)):
                    D_wj, D_wi_wj = get_cell_probability(data, gene, quantiles, beta_topk[j])
                    if D_wi_wj == 0:
                        f_wi_wj = 0
                    else:
                        f_wi_wj = (np.log2(D_wi_wj) - np.log2(D_wi) - np.log2(D_wj)) 
                        if self.coherence_norm:
                            f_wi_wj /= -np.log2(D_wi_wj)

                    TC_k[i, j] = f_wi_wj
                    TC_k[j, i] = f_wi_wj

            TC.append(TC_k[np.tril_indices_from(TC_k, k=-1)].mean())

        if individual:
            return TC
        return np.mean(TC)

    def get_topic_diversity(self):
        topics = self.topics
        num_topics = len(topics)
        genes = list(itertools.chain(*topics))
        n_unique = len(np.unique(genes))
        TD = n_unique / (self.topk * num_topics)
        return TD

    def get_topic_gene_cosine(self, individual=False):
        data = self.data 
        topics = self.topics
        TGC = []
        for i in range(len(topics)):
            topic_genes = topics[i]
            A = data[:, topic_genes]
            B = self.topic_prop[:, i]
            cos = (A.T @ B).flatten() / (norm(A.T, axis=1) * norm(B))
            TGC.append(cos.mean())

        if not individual:
            return np.mean(TGC)
        else:
            return TGC

    def get_topic_jaccard(self):
        from itertools import combinations
        topics = self.topics        
        sim = 0
        count = 0
        for list1, list2 in combinations(topics, 2):
            intersection = len(list(set(list1).intersection(list2)))
            union = (len(list1) + len(list2)) - intersection
            count = count + 1
            sim = sim + (float(intersection) / union)
        return sim / count

    def _prepare_kl_measure(self):
        def _preprocess_dist(arr):
            zero_lines = np.where(~arr.any(axis=1))[0]
            val = 1.0 / len(arr[0])
            vett = np.full(len(arr[0]), val)
            for zero_line in zero_lines:
                arr[zero_line] = vett.copy()
            arr = arr / arr.sum(axis=1, keepdims=True)
            return arr

        def _make_uniform(arr):
            val = 1.0 / len(arr[0])
            vett = np.full(len(arr[0]), val)
            return vett
        
        def _make_vacuous(phi, theta):
            vacuous = np.zeros(phi.shape[1])
            for topic in range(len(theta)):
                p_topic = theta[topic].sum()/len(theta[0])
                vacuous += phi[topic]*p_topic
            return vacuous

        phi = _preprocess_dist(self.beta)
        theta = _preprocess_dist(self.topic_prop).T

        unif_phi = _make_uniform(phi)
        unif_theta = _make_uniform(theta)
        vacuous = _make_vacuous(phi, theta)
        
        self.pq_map = {
            "U": (phi, unif_phi), 
            "V": (phi, vacuous),
            "B": (theta, unif_theta) 
        }

    def get_kl_metric(self, measure, individual=False):
        def _KL(P, Q):
            # add epsilon to grant absolute continuity
            epsilon = 0.00001
            P = P+epsilon
            Q = Q+epsilon

            divergence = np.sum(P*np.log(P/Q))
            return divergence
        if not hasattr(self, "pq_map"):
            self._prepare_kl_measure()
        assert measure in self.pq_map.keys()

        divergence = []
        p = self.pq_map[measure][0]
        q = self.pq_map[measure][1]
        for topics in range(len(p)):
            divergence.append(_KL(p[topics]/p[topics].sum(), q))
        if individual:
            return divergence
        else:
            result = np.mean(divergence)
            return result if not np.isnan(result) else 0

    def get_metrics_detail(
        self,
        topic_coherence=True,
        topic_diversity=True,
        rank_biased_overlap=True,
        topic_gene_cosine=True,
        topic_jaccard=True,
        topic_sparsity=True,
        kl_uniform=True,
        kl_vacuous=True,
        kl_background=True,
    ):
        metrics = {}
        if topic_coherence:
            metrics["Module Coherence"] = self.get_coherence_NPMI()
        if topic_diversity:
            metrics["Module Diversity"] = self.get_topic_diversity()
        if rank_biased_overlap:
            metrics["Module Diversity(RBO)"] = self.get_rbo()
        if topic_gene_cosine:
            metrics["Gene Topic Coherence"] = self.get_topic_gene_cosine()
        if topic_jaccard:
            metrics["Topic Jaccard"] = self.get_topic_jaccard()
        if topic_sparsity:
            metrics["Topic Sparsity"] = self.get_topic_sparseness()
        if kl_uniform:
            metrics["KL Divergence (Uniform)"] = self.get_kl_metric("U")
        if kl_vacuous:
            metrics["KL Divergence (Vacuous)"] = self.get_kl_metric("V")
        if kl_background:
            metrics["KL Divergence (Background)"] = self.get_kl_metric("B")
        return metrics

    def get_metrics(self):
        metrics = self.get_metrics_detail(
            topic_coherence=True,
            topic_diversity=True,
            rank_biased_overlap=False,
            topic_gene_cosine=True,
            topic_jaccard=False,
            topic_sparsity=False,
            kl_uniform=True,
            kl_vacuous=True,
            kl_background=True,
        )
        return {
            "Module Coherence": metrics["Module Coherence"],
            "Module Diversity": metrics["Module Diversity"],
            "Gene Topic Coherence": metrics["Gene Topic Coherence"],
            "Topic Significance": (
                metrics["KL Divergence (Uniform)"] + \
                metrics["KL Divergence (Vacuous)"] + \
                metrics["KL Divergence (Background)"]
            ) / 3,
        }
    
    
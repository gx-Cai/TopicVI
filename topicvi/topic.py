from .utils import array_nlargest
import numpy as np
import pandas as pd

class TopicDict():

    def __init__(self, topic_list):

        self.n_topics = len(topic_list)
        self.topic_list = topic_list

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.topic_list[idx]
        elif isinstance(idx, slice):
            return self.topic_list[idx]
        else:
            raise TypeError(f"Index must be int or slice, not {type(idx)}")

    def __len__(self):
        return self.n_topics
    
    def __str__(self):
        return f"TopicDict with {self.n_topics} topics"

    def search_gene(self, gene):
        for i, topic in enumerate(self.topic_list):
            if gene in topic:
                return i
        return -1

    @classmethod
    def transfer_from_adata(
        cls,
        adata,
        topk = 50,
        topic_comp = None,
        topic_comp_key = 'topic_by_gene',

    ):
        gene_names = adata.var_names
        if topic_comp is None:
            beta = adata.varm[topic_comp_key].T
        else:
            beta = topic_comp
        topics_idx = array_nlargest(beta, topk, vmin=0)
        topics_list = [gene_names[i] for i in topics_idx]
        return cls(
            topics_list,
        )
                
    def compare_prior_overlap(self, prior):
        if prior is None:
            raise ValueError("prior is None")
        if not isinstance(prior, dict):
            raise ValueError("prior should be a dict")
                
        overlap = np.zeros((len(prior), self.n_topics))
        # get the overlap number
        for i, key in enumerate(prior.keys()):
            si = set(prior[key])
            for j in range(self.n_topics):
                sj = set(self.topic_list[j])
                r = len(set(si) & set(sj)) / len(set(si) | set(sj))
                overlap[i, j] = r
        self.prior_overlap = overlap.T
        self.prior_annote = list(prior.keys())

    def __topic_annote(self, idx, ntop=3):
        if not hasattr(self, 'prior_overlap'):
            raise ValueError("run `compare_prior_overlap` first!")
        prior_idx = self.prior_overlap[idx, :].argsort()[-ntop:][::-1]
        annotes = [self.prior_annote[i] for i in prior_idx]
        return {
            annotes[i]: self.prior_overlap[idx, prior_idx[i]].tolist() for i in range(ntop)
        }
    
    def get_topic_annotes(self, ntop=3):
        if not hasattr(self, 'prior_overlap'):
            raise ValueError("run `compare_prior_overlap` first!")
        topic_annotes = []
        for i in range(self.n_topics):
            topic_annotes.append(self.__topic_annote(i, ntop))
        return topic_annotes
    
    def scan_topic_overlap(self, idx):
        if not hasattr(self, 'prior_overlap'):
            raise ValueError("run `compare_prior_overlap` first!")
        if idx >= self.n_topics | idx < 0:
            raise ValueError(f"index {idx} out of range")

        result = self.prior_overlap[idx, :]
        result = pd.Series(
            result,
            index=self.prior_annote,
            name=f"topic_{idx}",
        ).sort_values(ascending=False)
        result = result[result > 0]
        return result
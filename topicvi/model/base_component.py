import torch
from torch import nn as nn
from torch.distributions import Normal, Categorical
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
from scvi.nn import Encoder, Decoder
from .sinkhorn import sinkhorn_costmat
import numpy as np

def safe_exp(x, x_min=-10, x_max=10):
    return torch.exp(torch.clamp(x, x_min, x_max))

class TopicEncoder(Encoder):
    def __init__(
        self, 
        n_input: int, n_output: int, 
        n_cat_list = None, 
        n_layers: int = 1, 
        n_hidden: int = 128, 
        dropout_rate: float = 0, 
        distribution: str = "normal", 
        var_eps: float = 0.0001, 
        var_activation = None, 
        return_dist: bool = False,
        **kwargs
    ):
        super().__init__(n_input, n_output, n_cat_list, n_layers, n_hidden, dropout_rate, distribution, var_eps, var_activation, return_dist, **kwargs)

    def forward(self, x: torch.Tensor, *cat_list: int):
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q).clamp(-4.0, 4.0)/2) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent

class PriorTopicDecoder(nn.Module):
    def __init__(
        self, 
        n_genes, n_hidden, n_topics, 
        n_topics_without_prior = None,
        topic_similarity_penalty_weight = 1,
        nomalize=False, 
    ) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(n_hidden, n_genes), requires_grad=True)
        self.n_hidden = n_hidden
        self.topic_emb = nn.Parameter(
            torch.randn(n_topics, n_hidden), 
            requires_grad=True
        )
        self.tpw = topic_similarity_penalty_weight
        torch.nn.init.xavier_normal_(self.topic_emb)
        torch.nn.init.xavier_normal_(self.embedding)
        self.nomalize = nomalize
        self.n_topics = n_topics
        self.n_topics_without_prior = n_topics_without_prior if n_topics_without_prior is not None else int(n_topics * 0.25)
        assert self.n_topics_without_prior <= n_topics, '`n_topics_without_prior` should be less than n_topics'

    def load_gene_embedding(self, embedding, fix=False):
        assert embedding.shape[1] == self.embedding.shape[1], 'embedding shape should be [n_hidden, n_genes], found n_genes is not matched'
        if embedding.shape[0] != self.embedding.shape[0]:
            print("Warning: embedding hidden layer shape changed to {}".format(embedding.shape[0]))
            self.n_hidden = embedding.shape[0]
            self.topic_emb = nn.Parameter(
                torch.randn(self.n_topics, self.n_hidden).type_as(self.topic_emb), 
                requires_grad=True
            )
        # match dtypes
        embedding = embedding.type_as(self.embedding)
        self.embedding = nn.Parameter(embedding, requires_grad=not fix)
        return self.n_hidden

    def forward(self, z: torch.Tensor):
        beta = self.get_topics() 
        aa = torch.mm(z, beta)
        return aa
        
    def get_topics(self):
        topic_emb = self.get_topic_emb()
        beta = topic_emb @ self.embedding
        return F.relu(beta)

    def get_topic_emb(self):
        if self.nomalize:
            topic_emb = F.normalize(self.topic_emb, p=2, dim=-1)
        else:
            topic_emb = self.topic_emb
        return topic_emb # [n_hidden, topic_dim]

    def get_cost_matrix(self, index):
        if not index or (self.n_topics_without_prior == self.n_topics):
            return None
        topics = self.get_topics()[0:self.n_topics-self.n_topics_without_prior, :]
        m = - torch.log_softmax(topics, dim =-1) # [n_hidden, n_genes]
        cost_matrix = torch.cat([m[:, i].mean(axis=1).reshape(1, -1) for i in index]) # [n_hidden, n_index]
        return cost_matrix # [n_hidden, n_index]

    def semisupervised_topic_loss(self, cost_matrix, weight = 100, **kwargs):
        if cost_matrix is None:
            return self.topic_similarity_penalty()
        loss = sinkhorn_costmat(cost_matrix, **kwargs) * np.log1p(cost_matrix.numel()) * weight
        return loss + self.topic_similarity_penalty()
    
    def topic_similarity_penalty(self, weight=1000):
        def cosine_similarity_single(topic_emb):
            if topic_emb.shape[0] <= 1:
                return 0.0
            m = topic_emb / (torch.norm(topic_emb, dim=-1, keepdim=True) + 1e-12) # [n_hidden, n_input]
            cosine = (m @ m.transpose(0, 1)).abs()
            mean = cosine.mean()
            var = ((cosine - mean) ** 2).mean()
            return mean - var

        if self.tpw is None:
            return 0.0
        
        topic_emb = self.get_topics()
        loss_div = 0
        if self.n_topics_without_prior == self.n_topics:
            loss_div = cosine_similarity_single(topic_emb) * self.n_topics
        else:
            topic_emb1 = topic_emb[-self.n_topics_without_prior:, :]
            # topic_emb2 = topic_emb[:-self.n_topics_without_prior, :]
            loss_div = (
                cosine_similarity_single(topic_emb1) * self.n_topics_without_prior / self.n_topics + 
                cosine_similarity_single(topic_emb) * (self.n_topics - self.n_topics_without_prior) / self.n_topics
            )
        
        return loss_div * weight * self.tpw

class ClusterTopicDecoder(nn.Module):
    def __init__(
        self,
        n_latent: int,
        n_topics: int, 
        cluster_centers = None,
        cluster_number: int = None,
        alpha: float = 10,
        adaptive_penalty_weight = 1.0,
        selftraining_penalty_weight = 1.0,
        center_penalty_weight = 1.0,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterTopicDecoder, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha
        self.cluster_number = cluster_number
        self.n_topics = n_topics
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                cluster_number, n_latent, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)
        ## balancing the loss
        self.apw = adaptive_penalty_weight * 200
        self.spw = selftraining_penalty_weight * 50
        self.cpw = center_penalty_weight / 2
        self.inner_distance_weight = 15

        self._pu_c = nn.Parameter(
            torch.ones(cluster_number, n_topics),
            requires_grad=True
        )
        nn.init.xavier_normal_(self.pu_c)

        self.decoder = Decoder(
            n_topics,
            n_latent,
            n_cat_list=None,
            n_layers=1,
            use_layer_norm=True,
        )

        self.history_ = {
            'selftraining_loss': [],
            'adaptive_loss': [],
            'center_loss': [],
        }
        
    @property
    def pu_c(self):
        return self._pu_c.abs()

    def forward(self, batch, likelihood=None):
        act_func = nn.Softmax(dim=-1)
        distance = self.get_distance(batch)
        prob = self.prob_from_distance(distance) # [batch size, number of clusters]
        qu_z = prob @ self.pu_c # [batch size, n_topics]
        pz_m, pz_v = self.decoder(qu_z)
        pz = Normal(pz_m, pz_v.sqrt())
        if likelihood is None:
            wdist = distance
        else:
            w = self.get_latent_weight(batch, prob, likelihood)
            wdist = self.get_distance(batch, w)
        return act_func(qu_z), pz, wdist
    
    def get_latent_weight(self, latent, prob, likelihood):
        w = latent.T @ prob @ self.pu_c[:, 0:likelihood.shape[0]] @ likelihood # [n_latent, n_prior]
        return torch.softmax(w, dim=-1).mean(1)
        # return w.abs().mean(1)
    
    def observe_distribution(self, distance):
        norm_squared = distance.pow(2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self, q_ij):
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (q_ij ** 2) / torch.sum(q_ij, 0)
        return (weight.t() / torch.sum(weight, 1)).t()
    
    def get_distance(self, batch, w=None):
        """
        Compute the distance between each sample and each cluster center.

        :param batch: [batch size, embedding dimension] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        if w is None:
            return torch.norm(batch.unsqueeze(1) - self.cluster_centers, dim=2)
        else:
            return torch.sqrt((w * (batch.unsqueeze(1) - self.cluster_centers).pow(2)).sum(2))

    def prob_from_distance(self, distance):
        # return torch.exp(-distance) # [batch size, number of clusters]
        return torch.softmax(-distance, dim=-1)

    def get_prob(self, batch):
        return self.prob_from_distance(self.get_distance(batch))

    def get_assignment(self, batch,):
        """
        Compute the assignment for each sample.

        :param batch: [batch size, embedding dimension] Tensor of dtype float
        :return: [batch size] Tensor of dtype long
        """
        return torch.argmin(self.get_distance(batch), dim = 1)

    def adapative_loss(self, distance):
        prob = self.prob_from_distance(distance)
        adp_loss = (distance * prob).mean(1).mean() * self.apw
        return adp_loss

    def selftraining_loss(self, distance):
        x_hat = self.observe_distribution(distance)
        target = self.target_distribution(x_hat).detach()
        loss = self.kl_loss(x_hat.log(), target)
        return loss.sum() * self.spw

    def center_loss(self, distance):
        # the David-Bouldin index like loss
        distance2 = distance.pow(2) # [batch size, number of clusters]
        # S = torch.sqrt(torch.quantile(distance,q = 1 - 1/self.cluster_number, dim=0)) # [n_clusters]
        assignment = torch.argmin(distance2, dim=1)
        S = torch.zeros(self.cluster_number, device=distance.device)
        for i in range(self.cluster_number):
            iassign = assignment == i
            S[i] = torch.sqrt(distance2[iassign, i].mean())
        S[torch.isnan(S)] = 0
        S *= self.inner_distance_weight
        
        cdistance = torch.norm(self.cluster_centers - self.cluster_centers.unsqueeze(1), dim=2)
        loss = (S.unsqueeze(0) + S.unsqueeze(1)) - cdistance
        loss = loss.sum() / self.cluster_number / (self.cluster_number - 1)
        return loss * self.cpw # .max(dim=-1).values

    def cluster_loss(self, distance):
        loss1 = self.selftraining_loss(distance)
        loss2 = self.adapative_loss(distance)
        loss3 = self.center_loss(distance)
        self.history_['selftraining_loss'].append(loss1.item())
        self.history_['adaptive_loss'].append(loss2.item())
        self.history_['center_loss'].append(loss3.item())
        return loss1 + loss2 + loss3
    
    def classification_loss(self, x, y):
        logit = -self.get_distance(x)
        loss = F.cross_entropy(logit, y.squeeze().long(), reduction='none')
        pt = torch.exp(-loss)
        loss = (1 - pt) ** 2 * loss

        # logprob = self.get_logprob(x)
        # loss = F.nll_loss(logprob, y.squeeze())
        return logit, loss.mean()
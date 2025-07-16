import scanpy as sc
from scanpy import AnnData
from typing import Literal
import warnings
import numpy as np
import pandas as pd
import torch
from torch.distributions import Normal, Categorical
from torch.distributions import kl_divergence as kl
import torch.nn.functional as F
import scvi
from scvi import REGISTRY_KEYS
from scvi.model._scvi import SCVI
from scvi.module.base import LossOutput, auto_move_data
from scvi.module._vae import VAE
from scvi.model._utils import _init_library_size
from scvi.data._utils import _get_adata_minify_type, _is_minified, get_anndata_attribute
from scvi.data.fields import (
    BaseAnnDataField,
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)
from scvi.data import AnnDataManager
from scvi.train import SemiSupervisedTrainingPlan, TrainRunner
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.train._callbacks import SubSampleLabels
from scvi.nn import Decoder, Encoder

from . import run_default_cluster
from .base_component import PriorTopicDecoder, TopicEncoder, ClusterTopicDecoder

def check_value_valid(value):
    import torch

    if torch.isnan(value).any() or torch.isinf(value).any():
        return False
    return True

def broadcast_labels(o, n_broadcast=-1):
    """Utility for the semi-supervised setting.

    If y is defined(labelled batch) then one-hot encode the labels (no broadcasting needed)
    If y is undefined (unlabelled batch) then generate all possible labels (and broadcast other
    arguments if not None)
    """
    ys_ = torch.nn.functional.one_hot(
        torch.arange(n_broadcast, device=o.device, dtype=torch.long), n_broadcast
    )
    ys = ys_.repeat_interleave(o.size(-2), dim=0)
    if o.ndim == 2:
        new_o = o.repeat(n_broadcast, 1)
    elif o.ndim == 3:
        new_o = o.repeat(1, n_broadcast, 1)
    return ys, new_o

class TopicVAE(VAE):
    def __init__(
        self,
        n_input: int,
        prior_genesets = None, 
        cluster_prior_genesets = None, 
        mode = 'unsupervised',
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        topic_decoder = None,
        cluster_decoder = None,
        n_continuous_cov: int = 0,
        n_cats_per_cov = None,
        dropout_rate: float = 0,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        log_variational: bool = True,
        gene_likelihood: Literal["zinb", "nb"] = "zinb",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "encoder",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        **vae_kwargs,
    ):
        super().__init__(
            n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            n_batch=n_batch,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **vae_kwargs,
        )

        self.prior = cluster_prior_genesets + prior_genesets
        self.n_cluster_prior = len(cluster_prior_genesets)
        self.topic_decoder = topic_decoder
        self.cluster_decoder = cluster_decoder
        self.n_labels = self.cluster_decoder.cluster_number

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"


        self.z_encoder = TopicEncoder(
            n_input,
            n_latent,
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )
        self.mode = mode
        
        self.encoder_z2_z1 = Encoder(
            n_latent,
            self.cluster_decoder.n_topics,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            return_dist=True,
        )

        self.decoder_z1_z2 = Decoder(
            self.cluster_decoder.n_topics,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )


    @auto_move_data
    def loss(
        self,
        tensors, inference_outputs, generative_outputs,
        # --- arguments for semi-supervised training --- #
        labelled_tensors: dict[str, torch.Tensor] = None,
        classification_ratio: float = 1.0,
        feed_labels: bool = False,
        # --- arguments for loss computation --- #
        kl_weight: float = 1.0,
        ce_weight = 1.0,
        cl_weight = 1.0,
        ce_loss_kwargs = dict(),
    ):
        
        if feed_labels:
            warnings.warn("feed_labels is not supported in TopicVAE, will be ignored", category=RuntimeWarning)

        x = tensors[REGISTRY_KEYS.X_KEY]
        qz1 = inference_outputs["qz"]
        z1 = inference_outputs["z"]
        px = generative_outputs['px']
        optmat = self.topic_decoder.get_cost_matrix(index = self.prior,)
        if optmat is None or self.n_cluster_prior == 0:
            qu_z, pz2, distance = self.cluster_decoder(z1, None)
        elif self.n_cluster_prior > 0:
            qu_z, pz2, distance = self.cluster_decoder(
                z1, torch.exp(-optmat[:, 0:self.n_cluster_prior].T)
            )
        px_alpha = self.topic_decoder(qu_z)
        
        ce_loss = self.topic_decoder.semisupervised_topic_loss(optmat, **ce_loss_kwargs)
        cl_loss = self.cluster_decoder.cluster_loss(distance)
        probs = self.cluster_decoder.prob_from_distance(distance) # [n_obs, n_clusters]
        
        ys, z1s = broadcast_labels(z1, n_broadcast=self.n_labels)
        qz2, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
        mean = torch.zeros_like(qz2.loc)
        scale = torch.ones_like(qz2.scale)
        kl_divergence_z2 = kl(qz2, Normal(mean, scale)).sum(dim=-1)
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = qz1.log_prob(z1).sum(dim=-1)
        if z1.ndim == 2:
            loss_z1_unweight_ = loss_z1_unweight.view(self.n_labels, -1).t()
            kl_divergence_z2_ = kl_divergence_z2.view(self.n_labels, -1).t()
        else:
            loss_z1_unweight_ = torch.transpose(
                loss_z1_unweight.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
            kl_divergence_z2_ = torch.transpose(
                kl_divergence_z2.view(z1.shape[0], self.n_labels, -1), -1, -2
            )
        
        dir_reconstruction_loss = (torch.log_softmax(px_alpha, dim=-1) * px.get_normalized('mu')).sum(dim=-1)
        reconstruction_loss = -(
            px.log_prob(x).sum(dim=-1) + \
            dir_reconstruction_loss 
            + qz2.log_prob(qu_z.repeat(self.n_labels, 1)).sum(dim=-1).view(-1, self.n_labels).sum(dim=-1)
        ) + loss_z1_weight + (loss_z1_unweight_ * probs).sum(dim=-1)

        kl_local = (kl_divergence_z2_ * probs).sum(dim=-1)
        kl_divergence_l = self.get_local_library_kl_loss(tensors, inference_outputs)
        # cluster assignment
        kl_divergence_c = kl(
            Categorical(probs), 
            Categorical(torch.ones_like(probs) / probs.shape[1])
        )
        kl_local += kl_divergence_l + kl_divergence_c.sum(dim=0) + kl(qz1, pz2).sum(dim=1)
        kl_local = kl_local.mean()

        classify_loss = 0.0
        acc = 0.0
        if labelled_tensors is not None and self.mode != 'unsupervised':
            ly = labelled_tensors['labels_']  # (n_obs, 1) REGISTRY_KEYS.LABELS_KEY
            l_inference_inputs = self._get_inference_input(labelled_tensors)
            l_inference_outputs = self.inference(**l_inference_inputs)
            lqz = l_inference_outputs["qz"]
            logit, classify_loss = self.cluster_decoder.classification_loss(lqz.loc, ly)
            acc = (logit.argmax(dim=1) == ly.squeeze(-1)).float().mean()
            cl_loss += classification_ratio * classify_loss
        
        loss = torch.mean(
            reconstruction_loss + kl_weight * kl_local + \
            ce_loss * ce_weight + cl_weight * cl_loss
        )
        if not check_value_valid(loss):
            assert check_value_valid(reconstruction_loss)
            assert check_value_valid(kl_local)
            assert check_value_valid(ce_loss)
            assert check_value_valid(cl_loss)
            assert check_value_valid(classify_loss)
    
        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=kl_local,
            extra_metrics=dict(
                ce_loss=ce_loss * ce_weight,
                dir_reconstruction_loss=-dir_reconstruction_loss.mean(),
                cl_loss=cl_weight*cl_loss, 
                classify_loss=classification_ratio * classify_loss,
                acc = acc,
            )
        )

    def get_local_library_kl_loss(self, tensors, inference_outputs):
        if not self.use_observed_lib_size:
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            ql = inference_outputs["ql"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                ql,
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = 0.0
        return kl_divergence_l

    def get_weights_from_pretrained_scvi_model(
        self,
        scvi_model,
        generative_z: torch.Tensor, # (n_obs, n_latent)
        labels
    ):
        """
        Copy the weights from a pre-trained scvi model to the current model.
        Only z_encoder, decoder and l_encoder are copied. And initial the cluster_mean with generative z.
        """
        
        self.z_encoder.load_state_dict(scvi_model.z_encoder.state_dict())
        self.decoder.load_state_dict(scvi_model.decoder.state_dict())
        self.l_encoder.load_state_dict(scvi_model.l_encoder.state_dict())
        centers = self.get_cluster_centers_by_labels(generative_z, labels) # [n_clusters, n_latent]
        self.cluster_decoder.cluster_centers.weight = centers
        return self

    def get_cluster_centers_by_labels(self, generative_z, labels):
        labels = labels.squeeze(-1) #tensors[REGISTRY_KEYS.LABELS_KEY]
        centers = torch.stack(
            [
                generative_z[labels == i].mean(dim=0) 
                for i in range(self.cluster_decoder.cluster_number)
            ]
        )
        return centers.detach()


class TopicVI(SCVI):

    __topic_decoder_csl = PriorTopicDecoder
    __cluster_decoder_csl = ClusterTopicDecoder
    __module_cls = TopicVAE

    def __init__(
        self,
        adata,
        n_labels = None,
        n_topics = None,
        prior_genesets = None,
        cluster_prior_genesets = None,
        topic_decoder_params = {},
        cluster_decoder_params = {},
        n_hidden: int = 128,
        n_latent: int = 32,
        n_layers: int = 2,
        dropout_rate: float = 0,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb"] = "zinb",
        mode: Literal["supervised", "unsupervised", "semisupervised"] = 'unsupervised',
        **model_kwargs,
    ):
        super().__init__(adata,)

        if dispersion == 'gene-label' and mode == 'unsupervised':
            warnings.warn("Label will also be used in initialization of cell cluster inference but will not simutaneously updated in topic decoder, unconsidered influence may occur. Please be cautious.", category=RuntimeWarning)

        n_batch = self.summary_stats.n_batch
        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key and self.minified_data_type is None:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)
        
        self.n_vars = self.summary_stats.n_vars
        # self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY).cat.categories.size #
        self.n_topics = n_topics if n_topics is not None else (len(cluster_prior_genesets) + len(prior_genesets))*2
        if self.n_topics > 1000:
            warnings.warn("The number of topics is too large, it may make traning for a long time.", category=RuntimeWarning)
        self.mode = mode

        prior_name, prior_genesets = self.__prior_genesets_preprocess(prior_genesets)
        cluster_prior_name, cluster_prior_genesets = self.__prior_genesets_preprocess(cluster_prior_genesets)
        self.prior_name = cluster_prior_name + prior_name
        self.n_cluster_prior = len(cluster_prior_name)
        prior_index_transfered = self.convert_genesets2idx(prior_genesets)
        cluster_prior_index_transfered = self.convert_genesets2idx(cluster_prior_genesets)
        if mode == 'supervised':
            self.n_labels = np.unique(self.adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)).size
            if self.n_labels != n_labels:
                warnings.warn("The number of labels is not equal to the number of prior genesets, will use the number of labels instead.", category=RuntimeWarning)
        else:
            self.n_labels = n_labels if n_labels is not None else self.n_cluster_prior

        params = {"n_hidden": n_hidden, 'n_topics_without_prior': int(self.n_topics*0.25)}
        params.update(topic_decoder_params)
        topic_decoder = self.__topic_decoder_csl(
            n_topics=self.n_topics,
            n_genes=self.n_vars,
            **params
        )

        cluster_decoder = self.__cluster_decoder_csl(
            n_latent = n_latent,
            n_topics=self.n_topics,
            cluster_centers = None,
            cluster_number = self.n_labels,
            **cluster_decoder_params
        )

        self.module = self.__module_cls(
            n_input=self.summary_stats["n_vars"],
            topic_decoder=topic_decoder,
            cluster_decoder=cluster_decoder,
            prior_genesets=prior_index_transfered,
            cluster_prior_genesets=cluster_prior_index_transfered,
            n_batch=self.summary_stats["n_batch"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            mode=mode,
            use_size_factor_key=use_size_factor_key,
            **model_kwargs,
        )
        self.module.minified_data_type = self.minified_data_type
        self.init_params_ = self._get_init_params(locals())

        self._model_summary_string = (
            "TopicVI model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}."
            f"n_topics: {self.n_topics}, n_labels: {self.n_labels}, mode: {mode}"
        )

        self._set_indices_and_labels()
        self.pretrained_ = False

    # --- process prior genesets --- #

    def convert_genesets2idx(self, prior_genesets):
        def convert_genesets_step(var_names, prior_genesets):
            prior_index_transfered = []
            for i in prior_genesets:
                if i not in var_names:
                    continue
                prior_index_transfered.append(var_names.get_loc(i))
            if (n_miss := (len(prior_genesets) - len(prior_index_transfered))) > 0:
                warnings.warn(f"{n_miss} genes is not in the gene list, will be ignored")
            return prior_index_transfered
        
        if prior_genesets is None:
            prior_index_transfered = None
        else:
            prior_index_transfered = [
                convert_genesets_step(self.adata.var_names,gs) for gs in prior_genesets
            ]
            
        return [i for i in prior_index_transfered if len(i) > 0]

    def __prior_genesets_preprocess(self, prior):
        if type(prior) is list:
            prior_name = [f'prior_{i}' for i, gs in enumerate(prior)]
        elif type(prior) is dict:
            prior_name = list(prior.keys())
            prior = list(prior.values())
        else:
            raise ValueError("prior should be a list or a dict")
        return prior_name, prior
    
    # not recommend to use
    def load_gene_embedding(self, embedding, gene_names, fix=False):
        adata = self.adata
        if not adata.var_names.isin(gene_names).all():
            warnings.warn(
                "Some genes are not in the adata, will be fill with random value"
                "In this case, recommment to set fix=False"
                "To avoid the warning, may check the gene_names and adata.var_names. (Gene names may be Alias)", 
                category=RuntimeWarning)
            missing_genes = adata.var_names[~adata.var_names.isin(gene_names)]
            embedding = np.concatenate(
                [embedding, np.random.rand(len(missing_genes), embedding.shape[1])], axis=0
            )
            gene_names = np.concatenate([gene_names, missing_genes])

        gene_idx = [i for i, g in enumerate(gene_names) if g in adata.var_names]
        embedding = embedding[gene_idx, :]
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        embedding = torch.from_numpy(embedding.T).to(self.device)
        n_hidden = self.module.topic_decoder.load_gene_embedding(embedding, fix)
        self.init_params_['non_kwargs']['topic_decoder_params'].update({'n_hidden': n_hidden})
        return self

    # --- setup and train --- #

    @classmethod
    def setup_anndata(
        cls, 
        adata: AnnData, 
        unlabeled_category = 'Unknown',
        layer = None, 
        batch_key = None, 
        labels_key = None, 
        size_factor_key = None, 
        categorical_covariate_keys = None, 
        continuous_covariate_keys = None, 
        run_cluster_kwargs = {},
        **kwargs
    ):
        if labels_key is None:
            warnings.warn("Label is not provided, falling back to default clustering method.")
            run_default_cluster(adata, layer, batch_key, **run_cluster_kwargs)
            labels_key = "default_cluster"
        
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField('labels_', labels_key, unlabeled_category), 
            # Unknown Bugs when set to REGISTRY_KEYS.LABELS_KEY
            LabelsWithUnlabeledObsField(REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
        ]
        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = None,
        n_samples_per_label: float = None,
        check_val_every_n_epoch: int = None,
        train_size: float = 0.9,
        validation_size: float = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        accelerator: str = "auto",
        devices: str = "auto",
        datasplitter_kwargs: dict = None,
        plan_kwargs: dict = None,
        warn_only = False,
        **trainer_kwargs,
    ):
        
        if not self.pretrained_:
            warnings.warn(
                "The model is not pretrained, it is recommended to pretrain the model before training."
                "Run model.pretrain() to pretrain the model.", 
                category=RuntimeWarning
            )
            if not warn_only:
                raise ValueError(
                    "The model is not pretrained, it is recommended to pretrain the model before training."
                    "Run model.pretrain() to pretrain the model."
                    "Or set warn_only=True to ignore the warning."
                )
            
        if self.module.mode == 'unsupervised':
            return super().train(
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices,
                train_size=train_size,
                validation_size=validation_size,
                shuffle_set_split=shuffle_set_split,
                batch_size=batch_size,
                datasplitter_kwargs=datasplitter_kwargs,
                plan_kwargs=plan_kwargs,
                check_val_every_n_epoch=check_val_every_n_epoch,
                **trainer_kwargs,
            )

        from scvi.model._utils import get_max_epochs_heuristic
        
        if max_epochs is None:
            max_epochs = get_max_epochs_heuristic(self.adata.n_obs)
        n_samples_per_label = n_samples_per_label or batch_size // self.n_labels
        plan_kwargs = {} if plan_kwargs is None else plan_kwargs
        datasplitter_kwargs = datasplitter_kwargs or {}

        # if we have labeled cells, we want to subsample labels each epoch
        sampler_callback = [SubSampleLabels()] if len(self._labeled_indices) != 0 else []

        data_splitter = SemiSupervisedDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            n_samples_per_label=n_samples_per_label,
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = SemiSupervisedTrainingPlan(self.module, self.n_labels, **plan_kwargs)
        if "callbacks" in trainer_kwargs.keys():
            trainer_kwargs["callbacks"] + [sampler_callback]
        else:
            trainer_kwargs["callbacks"] = sampler_callback

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **trainer_kwargs,
        )
        return runner()

    # --- inference or get results --- #

    def store_topics_info(self, ):
        topics_comp = self.get_topic_by_genes() # [n_topics, n_genes]
        topics_prop = self.get_topic_by_sample() # [n_obs, n_topics]
        self.adata.obsm['topic_by_sample'] = topics_prop.values
        self.adata.varm['topic_by_gene'] = topics_comp.T

    def get_topic_by_genes(self):
        return self.module.topic_decoder.get_topics().detach().cpu().numpy()
    
    def get_topic_by_sample(self, scale = True, adata = None):
        z = self.get_latent_representation(adata) # [n_obs, n_latent]
        z = torch.from_numpy(z).to(self.device)
        prob = torch.exp(self.module.cluster_decoder.get_prob(z)) # [n_obs, n_clusters]
        pu_c = self.module.cluster_decoder.pu_c.data.to(self.device) # [n_clusters, n_topics]
        align = prob @ pu_c #@ torch.exp(cost_matrix).T # [n_obs, n_index]
        if scale:
            align = ((align - align.min(axis=0).values) / (align.max(axis=0).values - align.min(axis=0).values))
        align = align.detach().cpu().numpy()
        align = pd.DataFrame(
            align, 
            columns=[f'topic_{i}' for i in range(align.shape[1])],
            index=self.adata.obs_names
        )
        return align

    @torch.inference_mode()
    def get_cluster_assignment(self, adata = None, max_distance_quantile = 0.99):
        if adata is None: adata = self.adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata)

        distances = []
        n_cluster_prior = self.module.n_cluster_prior
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            qz = inference_outputs["qz"]
            qz_m = qz.loc
            optmat = self.module.topic_decoder.get_cost_matrix(index = self.module.prior)
            
            if optmat is None or n_cluster_prior == 0:
                _0, _1, distance = self.module.cluster_decoder(qz_m, None)
            elif n_cluster_prior > 0:
                _0, _1, distance = self.module.cluster_decoder(
                    qz_m, torch.exp(-optmat[:, 0:n_cluster_prior].T)
                )
            distances.append(distance)

        distances = torch.cat(distances, dim=0)
        thres = distances.quantile(max_distance_quantile, dim=0)
        prob = self.module.cluster_decoder.prob_from_distance(distances)
        assignments = prob.argmax(dim=-1).detach()
        assignments[distances[torch.arange(distances.shape[0]), assignments] > thres[assignments]] = -1
        adata.obs['model_predict'] = assignments.cpu().numpy()
        adata.obs['model_predict'] = pd.Categorical(adata.obs['model_predict'])
        if adata.obs['model_predict'].cat.categories.size < self.n_labels:
            warnings.warn(
                "The number of clusters is not equal to the number of labels, some clusters may be ignored."
                "Try to adjust the cluster weight for expected results.",
                category=RuntimeWarning
            )

        return adata.obs['model_predict']
    
    def get_refined_cluster_assignment(
            self, 
            adata = None,
            minium_cluster_distance = None,
            q = 0.05,
            metrics_func = None, # accept embedding and pred
        ):
        from sklearn.metrics import davies_bouldin_score

        pred = self.get_cluster_assignment(adata)
        if adata is None: adata = self.adata
        metrics_func = metrics_func or davies_bouldin_score

        cluster_centers = self.module.cluster_decoder.cluster_centers.data.detach().cpu()
        cdistance = torch.norm(
            cluster_centers.unsqueeze(0) - cluster_centers.unsqueeze(1),
            dim=-1
        )
        if minium_cluster_distance is None:
            minium_cluster_distance = cdistance.quantile(q).item()
        search_pairs = torch.stack(
            torch.where(cdistance < minium_cluster_distance)
        ).T
        
        embed = self.get_latent_representation(adata)
        
        for ci, cj in search_pairs:
            if ci == cj: continue
            # greedy merge with db index
            prev_score = metrics_func(embed, pred)
            merged = pred.copy()
            merged[merged == ci] = cj
            score = metrics_func(embed, merged)
            if score < prev_score:
                pred = merged
                
        adata.obs['model_predict_refined'] = pred
        return adata.obs['model_predict_refined']

    # --- supervised mode --- #

    def _set_indices_and_labels(self):
        """Set indices for labeled and unlabeled cells."""
        labels_state_registry = self.adata_manager.get_state_registry('labels_') # REGISTRY_KEYS.LABELS_KEY
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping

        # set unlabeled and labeled indices
        self._unlabeled_indices = np.argwhere(labels == self.unlabeled_category_).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = dict(enumerate(self._label_mapping))

    def predict(self, adata=None, max_distance_quantile = 0.99):
        mapper = self._code_to_label
        mapper[-1] = 'Unassigned'
        label = self.get_cluster_assignment(adata, max_distance_quantile)
        return label.map(mapper)

    @classmethod
    def seed_label_from_topic(
        cls,
        adata, 
        annotation, 
        n_seed=None, 
        key_added='seed_label',
        unlabeled_category='Unknown',
        layer=None
    ):
        """
        Seed label from topic, modified from scanvi seed label toturial.
        """
        def get_score(normalized_adata, gene_set):
            """Returns the score per cell given a dictionary of + and - genes

            Parameters
            ----------
            normalized_adata
            anndata dataset that has been log normalized and scaled to mean 0, std 1
            gene_set
            a dictionary with two keys: 'positive' and 'negative'
            each key should contain a list of genes
            for each gene in gene_set['positive'], its expression will be added to the score
            for each gene in gene_set['negative'], its expression will be subtracted from its score

            Returns
            -------
            array of length of n_cells containing the score per cell
            """
            from scipy import sparse as sp
            score = np.zeros(normalized_adata.n_obs)
            if sp.issparse(expression := normalized_adata[:, list(gene_set["positive"])].X):
                expression = expression.toarray()
            score += expression.mean(axis=-1)
            if sp.issparse(expression := normalized_adata[:, list(gene_set["negative"])].X):
                expression = expression.toarray()
            score -= expression.mean(axis=-1)
            return score

        def get_cell_mask(normalized_adata, gene_set, n_cells = None):
            """Calculates the score per cell for a list of genes, then returns a mask for
            the cells with the highest 50 scores.

            Parameters
            ----------
            normalized_adata
            anndata dataset that has been log normalized and scaled to mean 0, std 1
            gene_set
            a dictionary with two keys: 'positive' and 'negative'
            each key should contain a list of genes
            for each gene in gene_set['positive'], its expression will be added to the score
            for each gene in gene_set['negative'], its expression will be subtracted from its score

            Returns
            -------
            Mask for the cells with the top 50 scores over the entire dataset
            """
            if n_cells is None:
                n_cells = normalized_adata.n_obs // 60
            score = get_score(normalized_adata, gene_set)
            cell_idx = score.argsort()[-n_cells:]
            mask = np.zeros(normalized_adata.n_obs)
            mask[cell_idx] = 1
            return mask.astype(bool)

        n_labels = len(annotation)
        if n_seed is None:
            n_seed = adata.n_obs // n_labels // 10
            n_seed = max(n_seed, 50)

        all_markers = set()
        cell_type_gene_markers = {}
        for ct in annotation.keys():
            cell_type_gene_markers[ct] = {}
            cell_type_gene_markers[ct]['positive'] = [i for i in annotation[ct] if i in adata.var_names]
            all_markers.update(cell_type_gene_markers[ct]['positive'])
        for ct in annotation.keys():
            cell_type_gene_markers[ct]['negative'] = [i for i in all_markers if i not in cell_type_gene_markers[ct]['positive']]

        adata.obs[key_added] = unlabeled_category
        normalized = adata.copy()
        if layer is not None:
            normalized.X = adata.layers[layer]
        
        cell_type_mask = {
            cell_type: get_cell_mask(normalized, gene_markers, n_cells=n_seed) 
            for cell_type, gene_markers in cell_type_gene_markers.items()
        }
        for cell_type, mask in cell_type_mask.items():
            adata.obs.loc[mask, key_added] = cell_type
        adata.obs[key_added] = adata.obs[key_added].astype('category')

    # --- pretrain --- #

    def pretrain(
        self,
        setup_kwargs = {}, 
        save = None,
        save_embedding = None,
        **train_kwargs
    ):
        # pretrain with SCVI model
        drop_params = ['topic_decoder_params', 'cluster_decoder_params', 'prior_genesets', 'embedding', 'cluster_prior_genesets', 'n_topics', 'mode', 'n_labels']
        parameters = self.init_params_['non_kwargs'].copy()
        extras = self.init_params_['kwargs']['model_kwargs'].copy()
        for i in drop_params:
            if i in parameters: 
                parameters.pop(i)
        parameters.update(extras)
        print('use the parameters for pretrain',parameters)
        adata = self.adata.copy()
        SCVI.setup_anndata(adata, **setup_kwargs)
        pretrain_model = SCVI(adata, **parameters)
        default_train_kwargs = dict(max_epochs=300, early_stopping=True, batch_size=128, plan_kwargs=dict(lr=1e-3, reduce_lr_on_plateau=True))
        default_train_kwargs.update(train_kwargs)
        pretrain_model.train(**default_train_kwargs)
        self.module.get_weights_from_pretrained_scvi_model(
            pretrain_model.module, 
            torch.from_numpy(pretrain_model.get_latent_representation()),
            torch.from_numpy(self.adata_manager.get_from_registry('labels_')) #REGISTRY_KEYS.LABELS_KEY
        )
        if save:
            pretrain_model.save(dir_path = save, overwrite=True)
            if save_embedding is None:
                save_embedding = adata.n_obs > 10000 
            if save_embedding:
                embedding = pretrain_model.get_latent_representation()
                with open(f'{save}/pretrain_embeding.pkl', 'wb') as f:
                    torch.save(embedding, f)
        self.pretrained_ = True
        return pretrain_model
    
    def load_pretrained_model(self, model):
        if isinstance(model, str):
            pretrain_model = SCVI.load(dir_path = model, adata=self.adata)
        elif isinstance(model, SCVI):
            pretrain_model = model
        try:
            with open(f'{model}/pretrain_embeding.pkl', 'rb') as f:
                embedding = torch.load(f)
            assert embedding.shape[0] == self.adata.n_obs
            assert embedding.shape[1] == self.module.n_latent
        except:
            embedding = pretrain_model.get_latent_representation()
        self.module.get_weights_from_pretrained_scvi_model(
            pretrain_model.module, 
            torch.from_numpy(embedding),
            torch.from_numpy(self.adata_manager.get_from_registry('labels_')) #REGISTRY_KEYS.LABELS_KEY
        )
        self.pretrained_ = True
        return self

    # --- after train analysis --- #

    def score_cluster_topics(self, topics = None, cluster_keys = 'model_predict', **kwargs):
        if cluster_keys == 'model_predict':
            if 'model_predict' not in self.adata.obs:
                self.get_cluster_assignment()
        elif cluster_keys not in self.adata.obs:
            raise ValueError(f"cluster_keys {cluster_keys} is not in the obs")
        if self.module.prior is None and topics is None:
            raise ValueError("No prior genesets provided")
        
        data = self.adata.copy()
        
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)
        sc.pp.scale(data)

        if topics is None:
            topics = self.module.prior
            # revert to gene names
            topics = [[data.var_names[i] for i in gs] for gs in topics]
        if type(topics) is list:
            topics = {
                f'topic_{i}': gs for i, gs in enumerate(topics)
            }

        for i, gs in topics.items():
            sc.tl.score_genes(
                data, gs, 
                score_name=f'{i}_score', 
                use_raw=False,
                **kwargs
            )
        score = data.obs.loc[:,[f"{i}_score" for i in topics.keys()] + [cluster_keys]].groupby(cluster_keys).mean().values
        
        return score
        
    def get_topic_combined_data(self):
        factors = self.get_topic_by_genes()
        fdata = sc.AnnData(
            X = self.get_topic_by_sample(),
            obs = self.adata.obs,
            var = pd.DataFrame(
                index=[f'topic_{i}' for i in range(factors.shape[0])],
                data=factors,
                columns=self.adata.var_names
            )
        )
        fdata.obsm['topicvi'] = self.get_latent_representation()
        return fdata
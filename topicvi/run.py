import scanpy as sc
import numpy as np
import os
import sys
import logging
from datetime import datetime
import yaml
from copy import deepcopy

from .model.module import TopicVI
from .model import inverse_davies_bouldin_score
from .prior import clean_prior_dict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='torchmetrics')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pytorch') 


class MultiDestLogger:
    def __init__(self, filename, verbose=False):
        self.original_stdout = sys.stdout  # Store the original stdout
        self.terminal = self.original_stdout
        self.log_file = open(filename, "a", encoding="utf-8")
        self.filename = filename  # Store filename for potential reopening
        self.closed = False
        self.verbose = verbose
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            handlers=[logging.FileHandler(filename), logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger()
        self.buffer = ""

    def write(self, message):
        if self.closed:
            # Reopen file if closed and still being used
            self.log_file = open(self.filename, "a", encoding="utf-8")
            self.closed = False

        # Skip IPython's object representation outputs
        if message.strip().startswith("[{") and '"type":' in message:
            self.terminal.write(message)
            return

        # Handle the message
        self.buffer += message

        # Process complete lines
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            if line.strip():
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"{timestamp} - {line}"
                if self.verbose:
                    self.terminal.write(line + "\n")
                self.log_file.write(log_message + "\n")
                self.log_file.flush()

    def flush(self):
        if not self.closed and self.buffer.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - {self.buffer}"
            if self.verbose:
                self.terminal.write(self.buffer)
            self.log_file.write(log_message + "\n")
            self.log_file.flush()
        self.buffer = ""
        if not self.closed:
            self.terminal.flush()

    def close(self):
        self.flush()
        if not self.closed:
            self.log_file.close()
            self.closed = True
            # Restore original stdout
            sys.stdout = self.original_stdout

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RunningPipeline:
    def __init__(self, run_func, data, running_config = None):
        self.run_fun = run_func

        if isinstance(data, str):
            if data.endswith(".h5ad"):
                data = sc.read_h5ad(data)


        self.data = data
        if running_config is None:
            if hasattr(self.data, "uns") and "running_config" in self.data.uns:
                self.running_config = self.data.uns["running_config"]
            else:
                raise ValueError("running_config must be provided.")
        elif isinstance(running_config, str):
            with open(running_config, "r") as f:
                self.running_config = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(running_config, dict):
            # deep copy the running config
            self.running_config = deepcopy(running_config)
        else:
            raise ValueError("running_config must be a path to a yaml file or a dictionary.")
        
        self.train_kwargs = self.running_config["train_kwargs"]
        self.data_kwargs = self.running_config["data_kwargs"]
        self.model_kwargs = self.running_config["model_kwargs"]
        self.extra_kwargs = self.running_config["extra_kwargs"]
        if run_func.__name__ in self.extra_kwargs:
            self.extra_kwargs = self.extra_kwargs[run_func.__name__]
            self.train_kwargs.update(self.extra_kwargs.get("train_kwargs", dict()))
            self.data_kwargs.update(self.extra_kwargs.get("data_kwargs", dict()))
            self.model_kwargs.update(self.extra_kwargs.get("model_kwargs", dict()))
        self.func_name = run_func.__name__
        self.data_name = self.running_config["project_name"]
        save_dir = self.running_config["save_dir"]
        save_dir = os.path.join(
            save_dir,
            self.func_name,
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.log = os.path.join(self.save_dir, "log.txt")

    def __call__(
        self, 
        compressed=False, 
        verbose=False, 
        save_model=None,
        check_runned=True
    ):        
        if check_runned and os.path.exists(os.path.join(self.save_dir, "results.npz")):
            print("Results already exist. Skipping.")
            return
        # make print statements write to log file
        with MultiDestLogger(self.log, verbose=verbose) as logger:
            if save_model is True:
                save_model = os.path.join(self.save_dir, "model")
            elif not isinstance(save_model, str):
                save_model = None
            if save_model:
                os.makedirs(save_model, exist_ok=True)
            try:
                self.results = self.run_fun(
                    self.data, 
                    data_kwargs=self.data_kwargs, 
                    train_kwargs=self.train_kwargs, 
                    model_kwargs=self.model_kwargs,
                    save_model=save_model,
                )
                self.save(compressed)
            except Exception as e:
                print("ERROR INFO", str(e))
                logger.close()
                raise
                
    def save(self, compressed=False):
        save_dir = os.path.join(self.save_dir, "results.npz")
        if compressed:
            np.savez_compressed(
                save_dir,
                **self.results,
            )
        else:
            np.savez(
                save_dir,
                **self.results,
            )

        # store the running config
        # with open(os.path.join(self.save_dir, "running_config.yaml"), "w") as f:
        #     yaml.dump(self.running_config, f)


def topicvi(
    adata, 
    train_kwargs=dict(pretrain_model=None),
    data_kwargs=dict(),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):
    running_mode = model_kwargs.get('running_mode', 'unsupervised')
    topic_decoder_params = model_kwargs.get('topic_decoder_params', dict())
    cluster_decoder_params = model_kwargs.get('cluster_decoder_params', dict())
    n_topics = model_kwargs.get('n_topics', 20)
    n_clusters = model_kwargs.get('n_clusters', 10)
    topicvi_kwargs = model_kwargs.get('topicvi_kwargs', dict())
    pretrain_kwargs = model_kwargs.get('pretrain_kwargs', dict())
    max_init_cells = model_kwargs.get('max_init_cells', 10000)

    cell_type_key = data_kwargs.get('label_key')
    batch_key = data_kwargs.get('batch_key')
    annotation_key = data_kwargs.get('annotation_key')
    size_factor_key = data_kwargs.get('size_factor_key')
    setup_kwargs = data_kwargs.get('setup_kwargs', dict())

    if running_mode == 'unsupervised':
        if (default_cluster_key:=data_kwargs.get('default_cluster_key')) is None:
            from .model import cluster_optimal_resolution
            cluster_optimal_resolution(
                adata, 
                label_key=None, 
                cluster_key='default_cluster', 
                use_rep='X_pca',
                cluster_function=sc.tl.leiden,
                metric=inverse_davies_bouldin_score,
                resolutions=np.linspace(0.1, 1, 10)
            )
        else:
            adata.obs['default_cluster'] = adata.obs[default_cluster_key]
    
    TopicVI.setup_anndata(
        adata,
        batch_key=batch_key,
        labels_key=cell_type_key if running_mode != 'unsupervised' else 'default_cluster',
        size_factor_key=size_factor_key,
        run_cluster_kwargs = dict(
            max_cells = max_init_cells
        ),
        **setup_kwargs
    )

    annotation = adata.uns[annotation_key]

    model = TopicVI(
        adata,
        n_topics=n_topics, 
        n_labels=n_clusters,
        prior_genesets=clean_prior_dict(annotation['background'], adata),
        cluster_prior_genesets=clean_prior_dict(annotation['clusters'], adata),
        mode = running_mode,
        topic_decoder_params=topic_decoder_params,
        cluster_decoder_params=cluster_decoder_params,
        **topicvi_kwargs
    )

    pretrain_model=train_kwargs.pop('pretrain_model', None)
    try:
        model.load_pretrained_model(pretrain_model)
    except Exception as e:
        print(
            'No pretrained model found or something error occured' 
            'start training from scratch.', e
        )
        model.pretrain(
            save = pretrain_model,
            setup_kwargs=dict(
                batch_key = batch_key,
                labels_key = cell_type_key if running_mode != 'unsupervised' else None,
            ),
            **pretrain_kwargs
        )
    
    if gene_emb_dir:=train_kwargs.pop('gene_emb_dir', None):
        gene_embedding = np.load(gene_emb_dir, allow_pickle=True).tolist()
        model.load_gene_embedding(
            gene_embedding['gene_emb'],
            gene_embedding['gene_ids'],
            fix = False
        )

    model.train(**train_kwargs)
    model.store_topics_info()
    if save_model:
        model.save(save_model, overwrite=True, save_anndata=False)
    
    return {
        "loading": model.get_topic_by_sample(),
        "factors": model.get_topic_by_genes(),
        'embedding': model.get_latent_representation(),
        'labels': model.get_cluster_assignment(),
    }


def topicvi_denovo_finding(
    adata, 
    train_kwargs=dict(pretrain_model=None, gene_emb_dir=None),
    data_kwargs=dict(),
    model_kwargs=dict(n_topics=20),
    save_model=None,
):
    n_topics = model_kwargs.get('n_topics', 20)
    topic_decoder_params = model_kwargs.get('topic_decoder_params', dict())
    topic_decoder_params.update(dict(n_topics_without_prior=n_topics))
    model_kwargs.update(dict(topic_decoder_params=topic_decoder_params))

    return topicvi(
        adata, 
        train_kwargs=train_kwargs, 
        data_kwargs=data_kwargs, 
        model_kwargs=model_kwargs,
        save_model=save_model
    )


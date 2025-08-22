
__version__ = "0.1.0"
__all__ = ["TopicVI", "TopicMetrics", "TopicDict", "pp", "pl", 'run_topicvi']

from .model.module import TopicVI
from .metrics import TopicMetrics
from .topic import TopicDict
from . import prior
from . import preprocess as pp
from . import ploting as pl
from .run import RunningPipeline, topicvi, topicvi_denovo_finding
import os


def __run_topicvi(
    data, config, run_func,
    compressed=False, 
    verbose=False, 
    save_model=None,
    check_runned=True
):
    RunningPipeline(
        run_func=run_func,
        data=data,
        running_config=config,
    )(
        compressed=compressed, 
        verbose=verbose, 
        save_model=save_model,
        check_runned=check_runned
    )


def run_topicvi(
    data, config, 
    compressed=False, 
    verbose=False, 
    save_model=None,
    check_runned=True
):
    """
    Run the TopicVI model with the given data and configuration.

    Parameters:
        data: The input data for the model.
        config: Configuration settings for the model.
        compressed (bool): Whether to use compressed data. Default is False.
        verbose (bool): Whether to print detailed logs. Default is False.
        save_model: Path to save the trained model. Default is None.
        check_runned (bool): Whether to check if the model has already been run. Default is True.

    Returns:
        None
    """
    __run_topicvi(
        data=data, 
        config=config, 
        run_func=topicvi,
        compressed=compressed, 
        verbose=verbose, 
        save_model=save_model,
        check_runned=check_runned
    )


def run_topicvi_denovo_finding(
    data, config, 
    compressed=False, 
    verbose=False, 
    save_model=None,
    check_runned=True
):
    """
    Run the TopicVI model with the given data and configuration.

    Parameters:
        data: The input data for the model.
        config: Configuration settings for the model.
        compressed (bool): Whether to use compressed data. Default is False.
        verbose (bool): Whether to print detailed logs. Default is False.
        save_model: Path to save the trained model. Default is None.
        check_runned (bool): Whether to check if the model has already been run. Default is True.

    Returns:
        None
    """
    __run_topicvi(
        data=data, 
        config=config, 
        run_func=topicvi_denovo_finding,
        compressed=compressed, 
        verbose=verbose, 
        save_model=save_model,
        check_runned=check_runned
    )


def make_default_config(
    project_name,
    working_dir,
):
    default_config = {
        "data_kwargs": {
            "annotation_key": "annotation",
            "batch_key": "library",
            "default_cluster_key": "leiden_opt",
            "size_factor_key": "size_factor",
        },
        "description": 'this is a default config, please modify it if need',
        "extra_kwargs": {
            "topicvi": {
                "data_kwargs": {"label_key": None},
                "model_kwargs": {
                    "cluster_decoder_params": {"center_penalty_weight": 1},
                    "pretrain_kwargs": {
                        "batch_size": 128,
                        "early_stopping": True,
                        "max_epochs": 1000,
                        "plan_kwargs": {"lr": 0.001, "reduce_lr_on_plateau": True},
                    },
                },
                "train_kwargs": {
                    "max_epochs": 1000,
                    "plan_kwargs": {"cl_weight": 1},
                    "pretrain_model": os.path.join(working_dir, project_name, "pretrain_model"),
                },
            },
            "topicvi_denovo_finding": {
                "data_kwargs": {"label_key": None},
                "train_kwargs": {
                    "pretrain_model": os.path.join(working_dir, project_name, "pretrain_model")
                },
            },
        },
        "model_kwargs": {"n_clusters": 10, "n_topics": 32},
        "project_name": project_name,
        "save_dir": os.path.join(working_dir, project_name),
        "train_kwargs": {"batch_size": 1024, "early_stopping": True, "max_epochs": 500},
    }
    return default_config

from time import ctime
import os
import sys
import logging
import numpy as np
from scipy import sparse as sp
import yaml

def verbose_print(*args, verbose, print_time=True, level="INFO", **kwargs):
    """
    print if verbose is True
    """
    # loging
    logging_level = getattr(logging, level.upper(), None)

    logging.log(
        logging_level,
        msg=" ".join(args),
        # *args, **kwargs
    )

    if not verbose:
        return None
    if print_time:
        print(ctime(), end=" ")
    print(*args, **kwargs)


def embed_dataframe(
    df, button="Download the data", filename="dataframe.csv", **export_args
):
    import base64
    from IPython.display import HTML, display

    # encode the dataframe as a csv file
    data = base64.b64encode(df.to_csv(**export_args).encode("utf8")).decode("utf8")

    # create a download link with filename and extension
    link = f"""
    <a href="data:text/csv;base64,{data}" download="{filename}">
    {button}
    </a>
    """
    # display the link
    display(HTML(link))


def check_value_valid(value):
    import torch

    if torch.isnan(value).any() or torch.isinf(value).any():
        return False
    return True


def is_count_data(array):
    if sp.issparse(array):
        array = array.data
    int_converted = array.astype(int).astype(float)
    return np.all(array == int_converted)

def array_nlargest(arr, n, axis = 1, vmin=0):
    # get the indices of the n largest elements in arr, and the values should be larger than vmin
    indx = arr.argsort(axis = axis)[:, -n:].tolist()
    nvalid = np.sum(arr>vmin, axis=axis)
    # if the number of valid elements is less than n, return all the elements
    ret = []
    for ind, nval in zip(indx, nvalid):
        if nval < n:
            ret.append([ind[-i] for i in range(1, nval+1)])
        else:
            ret.append(ind)
    return ret


def load_config(path='../running_config.yaml'):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_results(data_id, method, base_dir = '../results/'):
    dir = os.path.join(base_dir, data_id, method, 'results.npz')
    return np.load(dir, allow_pickle=True)


def write_config(config, path='../running_config.yaml'):
    with open(path, "w") as f:
        yaml.dump(config, f)
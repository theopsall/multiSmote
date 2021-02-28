import numpy as np
import pandas as pd

def get_classes(y) -> int:
    """
    Get the total number of classes.

    Args:
        y : The labels list of the data.

    Returns:
        int: Number of total classes
    """

    return int(y.shape[1])


def get_sum_classes(y) -> list:
    """
    Get the number of samples per class

    Args:
        y : Labels of the data

    Returns:
        list: [description]
    """

    if isinstance(y, pd.DataFrame):
        return pd.DataFrame(y).sum().tolist()
    if isinstance(y, np.ndarray):
        return y.sum(axis=0)
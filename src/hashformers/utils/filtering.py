"""
Consolidated filtering utilities for hashformers.

This module provides a single implementation of filter_top_k to be used
across the codebase, following the DRY principle (HASH-011).
"""

from typing import List, TypeVar, Callable, Optional
import copy
import pandas as pd
import numpy as np

T = TypeVar('T')


def filter_top_k(
    input_df: pd.DataFrame,
    k: int,
    gold_field: str = "hashtag",
    score_field: str = "score",
    segmentation_field: str = "segmentation",
    fill: bool = False
) -> pd.DataFrame:
    """
    Filter the top k rows of the input dataframe for each group defined by gold_field.
    
    This is the canonical implementation used across the codebase, consolidating
    duplicate implementations from experiments/evaluation.py and beamsearch/data_structures.py.

    The function sorts the input_df by score_field in ascending order and retains 
    the first k rows for each group. If fill option is set to True, it also clones 
    the records based on the length of each group.

    Args:
        input_df (pd.DataFrame): The input dataframe to filter.
        k (int): The number of top records to retain for each group.
        gold_field (str, optional): The field used to define groups in the dataframe. 
            Defaults to "hashtag".
        score_field (str, optional): The field used to sort the dataframe. 
            Defaults to "score".
        segmentation_field (str, optional): The field used if the fill option is set to True. 
            Defaults to "segmentation".
        fill (bool, optional): Whether to clone the records based on the length of each group. 
            Defaults to False.

    Returns:
        pd.DataFrame: The filtered dataframe.
    
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz'],
        ...     'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xyz'],
        ...     'score': [0.1, 0.2, 0.3, 0.1, 0.5]
        ... })
        >>> result = filter_top_k(df, k=2, gold_field='hashtag')
        >>> len(result)  # 2 per group = 4 total
        4
    """
    df = copy.deepcopy(input_df)
    
    df = df\
        .sort_values(by=score_field, ascending=True)\
        .groupby(gold_field)\
        .head(k)

    if fill:
        df["group_length"] = df.groupby(gold_field)[segmentation_field].transform(len)
        df["group_length"] = df["group_length"] * -1 + k + 1
        len_array = df["group_length"].values
        
        df = df.drop(columns=["group_length"])
        records = np.array(df.to_dict("records"))
        cloned_records = list(np.repeat(records, len_array))
        df = pd.DataFrame(cloned_records)
        
        df = df\
            .sort_values(by=score_field, ascending=True)\
            .groupby(gold_field)\
            .head(k)

        length = df.groupby(gold_field).size().values
        assert (length == k).all()
    
    return df


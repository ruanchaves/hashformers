from typing import List, Union, Any
import pandas as pd
from dataclasses import dataclass

@dataclass
class WordSegmenterOutput:
    output: List[str]
    segmenter_rank: Union[pd.DataFrame, None] = None
    reranker_rank: Union[pd.DataFrame, None] = None
    ensemble_rank: Union[pd.DataFrame, None] = None

@dataclass
class HashtagContainer:
    hashtags: List[List[str]]
    hashtag_set: List[str]
    replacement_dict: dict

@dataclass
class TweetSegmenterOutput:
    output: List[str]
    word_segmenter_output: Any

@dataclass
class CascadeNode:
    word_segmenter: Any
    word_segmenter_kwargs: dict
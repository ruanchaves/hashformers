from typing import Union, Any
import pandas as pd
from dataclasses import dataclass

@dataclass
class WordSegmenterOutput:
    output: list[str]
    segmenter_rank: Union[pd.DataFrame, None] = None
    reranker_rank: Union[pd.DataFrame, None] = None
    ensemble_rank: Union[pd.DataFrame, None] = None

@dataclass
class HashtagContainer:
    hashtags: list[list[str]]
    hashtag_set: list[str]
    replacement_dict: dict

@dataclass
class TweetSegmenterOutput:
    output: list[str]
    word_segmenter_output: Any
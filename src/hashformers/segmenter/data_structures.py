from dataclasses import dataclass
from typing import Union

import pandas as pd


@dataclass
class WordSegmenterOutput:
    """Return selected segmentations and optional component rankings."""

    output: list[str]
    segmenter_rank: Union[pd.DataFrame, None] = None
    reranker_rank: Union[pd.DataFrame, None] = None
    ensemble_rank: Union[pd.DataFrame, None] = None
    fusion_method: str = "top2"

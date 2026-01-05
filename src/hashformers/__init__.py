"""
Hashformers - Neural Hashtag Segmentation using Transformers.

This package provides tools for segmenting hashtags using transformer-based
language models with beam search and optional re-ranking.
"""

from hashformers.segmenter import *
from hashformers.segmenter.auto import *
from hashformers.config import HashformersConfig

__all__ = [
    # Main API
    "TransformerWordSegmenter",
    "RegexWordSegmenter",
    "TweetSegmenter",
    "TwitterTextMatcher",
    "BaseWordSegmenter",
    # Configuration
    "HashformersConfig",
    # Data structures
    "WordSegmenterOutput",
    "TweetSegmenterOutput",
    "HashtagContainer",
]

__version__ = "0.1.0"
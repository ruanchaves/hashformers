"""
Twitter text utilities for hashformers.

This module provides internal implementations for Twitter text processing,
reducing dependency on the deprecated twitter-text-python package (HASH-016).
"""

import re
from typing import List

# Pre-compiled pattern for hashtag extraction
HASHTAG_PATTERN = re.compile(r'#(\w+)', re.UNICODE)


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text.
    
    This is an internal implementation that can be used as an alternative
    to the deprecated twitter-text-python package.
    
    Args:
        text: Input text containing hashtags.
    
    Returns:
        List of hashtag strings without the # prefix.
    
    Example:
        >>> extract_hashtags("Hello #world #python")
        ['world', 'python']
        >>> extract_hashtags("No hashtags here")
        []
        >>> extract_hashtags("#CamelCase #with_underscore #123number")
        ['CamelCase', 'with_underscore', '123number']
    """
    return HASHTAG_PATTERN.findall(text)


def extract_hashtags_with_prefix(text: str) -> List[str]:
    """Extract hashtags from text, keeping the # prefix.
    
    Args:
        text: Input text containing hashtags.
    
    Returns:
        List of hashtag strings with the # prefix.
    
    Example:
        >>> extract_hashtags_with_prefix("Hello #world #python")
        ['#world', '#python']
    """
    return ['#' + tag for tag in extract_hashtags(text)]


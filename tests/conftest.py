"""Shared pytest fixtures and configuration.

HASH-409: Improve Test Coverage - Phase 3
Shared fixtures for hashformers test suite.
"""

import pytest
import torch
import pandas as pd

# Check GPU availability once at module load
CUDA_IS_AVAILABLE = torch.cuda.is_available()


def _check_spacy_available():
    """Check if spaCy is installed."""
    try:
        import spacy
        return True
    except ImportError:
        return False


def _check_langchain_available():
    """Check if LangChain is installed."""
    try:
        from langchain_core.documents import Document
        return True
    except ImportError:
        return False


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available."""
    return CUDA_IS_AVAILABLE


@pytest.fixture
def sample_hashtags():
    """Sample hashtags for testing."""
    return [
        "helloworld",
        "machinelearning",
        "naturallanguageprocessing",
        "deeplearning"
    ]


@pytest.fixture
def sample_segmentations():
    """Sample expected segmentations."""
    return [
        "hello world",
        "machine learning",
        "natural language processing",
        "deep learning"
    ]


@pytest.fixture
def sample_prob_dict_data():
    """Sample probability dictionary data."""
    return {
        "hello world": 0.1,
        "helloworld": 0.5,
        "he lloworld": 0.3,
        "h elloworld": 0.4,
        "test case": 0.2,
        "testcase": 0.6
    }


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing filter functions."""
    return pd.DataFrame({
        'hashtag': ['abc', 'abc', 'abc', 'xyz', 'xyz', 'xyz'],
        'segmentation': ['a bc', 'ab c', 'abc', 'x yz', 'xy z', 'xyz'],
        'score': [0.1, 0.2, 0.3, 0.1, 0.2, 0.5]
    })


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing without loading real models."""
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Simple mock: return ASCII values of characters
            return [ord(c) for c in text]
        
        def decode(self, token_ids, skip_special_tokens=True):
            return ''.join(chr(i) for i in token_ids if 32 <= i < 127)
    
    return MockTokenizer()


@pytest.fixture
def mock_model(mock_tokenizer):
    """Mock model for testing without GPU."""
    class MockModel:
        def __init__(self):
            self.tokenizer = mock_tokenizer
        
        def get_probs(self, texts):
            # Return mock probabilities based on text length
            # Shorter texts (more spaces) get lower scores
            return [len(text.replace(" ", "")) / len(text) for text in texts]
    
    return MockModel()


# Skip decorators for convenience
skip_if_no_gpu = pytest.mark.skipif(
    not CUDA_IS_AVAILABLE,
    reason="GPU required for this test"
)

skip_if_no_spacy = pytest.mark.skipif(
    not _check_spacy_available(),
    reason="spaCy not installed"
)

skip_if_no_langchain = pytest.mark.skipif(
    not _check_langchain_available(),
    reason="LangChain not installed"
)


# Export skip markers
__all__ = [
    'CUDA_IS_AVAILABLE',
    'skip_if_no_gpu',
    'skip_if_no_spacy', 
    'skip_if_no_langchain',
]

"""Tests for configuration module.

HASH-409: Improve Test Coverage - Phase 1.3
Tests for hashformers.config
"""

import pytest
import json
from pathlib import Path
from hashformers.config import HashformersConfig


class TestHashformersConfigDefaults:
    """Test HashformersConfig default values."""

    def test_default_topk(self):
        """Default topk should be 20."""
        config = HashformersConfig()
        assert config.topk == 20

    def test_default_steps(self):
        """Default steps should be 13."""
        config = HashformersConfig()
        assert config.steps == 13

    def test_default_alpha(self):
        """Default alpha should be 0.222."""
        config = HashformersConfig()
        assert config.alpha == 0.222

    def test_default_beta(self):
        """Default beta should be 0.111."""
        config = HashformersConfig()
        assert config.beta == 0.111

    def test_default_device(self):
        """Default device should be 'cuda'."""
        config = HashformersConfig()
        assert config.device == 'cuda'

    def test_default_gpu_batch_size(self):
        """Default gpu_batch_size should be 1000."""
        config = HashformersConfig()
        assert config.gpu_batch_size == 1000

    def test_default_segmenter_model(self):
        """Default segmenter_model_name_or_path should be 'gpt2'."""
        config = HashformersConfig()
        assert config.segmenter_model_name_or_path == 'gpt2'

    def test_default_segmenter_type(self):
        """Default segmenter_model_type should be 'gpt2'."""
        config = HashformersConfig()
        assert config.segmenter_model_type == 'gpt2'

    def test_default_reranker_model(self):
        """Default reranker_model_name_or_path should be None."""
        config = HashformersConfig()
        assert config.reranker_model_name_or_path is None

    def test_default_reranker_type(self):
        """Default reranker_model_type should be 'bert'."""
        config = HashformersConfig()
        assert config.reranker_model_type == 'bert'


class TestHashformersConfigCustomValues:
    """Test HashformersConfig with custom values."""

    def test_custom_topk(self):
        """Custom topk should be set correctly."""
        config = HashformersConfig(topk=10)
        assert config.topk == 10

    def test_custom_steps(self):
        """Custom steps should be set correctly."""
        config = HashformersConfig(steps=5)
        assert config.steps == 5

    def test_custom_device(self):
        """Custom device should be set correctly."""
        config = HashformersConfig(device='cpu')
        assert config.device == 'cpu'

    def test_custom_alpha_beta(self):
        """Custom alpha and beta should be set correctly."""
        config = HashformersConfig(alpha=0.5, beta=0.3)
        assert config.alpha == 0.5
        assert config.beta == 0.3

    def test_custom_segmenter_model(self):
        """Custom segmenter model should be set correctly."""
        config = HashformersConfig(
            segmenter_model_name_or_path='distilgpt2',
            segmenter_model_type='gpt2'
        )
        assert config.segmenter_model_name_or_path == 'distilgpt2'
        assert config.segmenter_model_type == 'gpt2'

    def test_custom_reranker_model(self):
        """Custom reranker model should be set correctly."""
        config = HashformersConfig(
            reranker_model_name_or_path='bert-base-cased',
            reranker_model_type='bert'
        )
        assert config.reranker_model_name_or_path == 'bert-base-cased'
        assert config.reranker_model_type == 'bert'

    def test_multiple_custom_values(self):
        """Multiple custom values should all be set correctly."""
        config = HashformersConfig(
            topk=15,
            steps=10,
            alpha=0.3,
            beta=0.2,
            device='cpu',
            gpu_batch_size=500
        )
        assert config.topk == 15
        assert config.steps == 10
        assert config.alpha == 0.3
        assert config.beta == 0.2
        assert config.device == 'cpu'
        assert config.gpu_batch_size == 500


class TestHashformersConfigDict:
    """Test HashformersConfig dict conversion methods."""

    def test_to_dict(self):
        """to_dict should return dictionary with all fields."""
        config = HashformersConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d['topk'] == 20
        assert d['steps'] == 13
        assert d['alpha'] == 0.222
        assert d['device'] == 'cuda'

    def test_to_dict_custom_values(self):
        """to_dict should preserve custom values."""
        config = HashformersConfig(topk=15, device='cpu')
        d = config.to_dict()
        assert d['topk'] == 15
        assert d['device'] == 'cpu'

    def test_from_dict(self):
        """from_dict should create config from dictionary."""
        d = {'topk': 15, 'steps': 10, 'device': 'cpu'}
        config = HashformersConfig.from_dict(d)
        assert config.topk == 15
        assert config.steps == 10
        assert config.device == 'cpu'

    def test_from_dict_partial(self):
        """from_dict with partial dict should use defaults for missing fields."""
        d = {'topk': 15}
        config = HashformersConfig.from_dict(d)
        assert config.topk == 15
        assert config.steps == 13  # default

    def test_roundtrip_dict(self):
        """Config should survive dict roundtrip."""
        original = HashformersConfig(topk=25, alpha=0.5)
        d = original.to_dict()
        restored = HashformersConfig.from_dict(d)
        assert restored.topk == original.topk
        assert restored.alpha == original.alpha
        assert restored.steps == original.steps


class TestHashformersConfigJSON:
    """Test HashformersConfig JSON I/O methods."""

    def test_to_json(self, tmp_path):
        """to_json should write valid JSON file."""
        config = HashformersConfig(topk=25, alpha=0.5)
        filepath = tmp_path / "config.json"
        config.to_json(str(filepath))

        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data['topk'] == 25
        assert data['alpha'] == 0.5

    def test_from_json(self, tmp_path):
        """from_json should load config from JSON file."""
        filepath = tmp_path / "config.json"
        with open(filepath, 'w') as f:
            json.dump({'topk': 30, 'steps': 8}, f)

        config = HashformersConfig.from_json(str(filepath))
        assert config.topk == 30
        assert config.steps == 8

    def test_json_roundtrip(self, tmp_path):
        """Config should survive JSON roundtrip."""
        original = HashformersConfig(
            topk=25,
            alpha=0.5,
            device='cpu',
            segmenter_model_name_or_path='distilgpt2'
        )
        filepath = tmp_path / "config.json"
        original.to_json(str(filepath))
        restored = HashformersConfig.from_json(str(filepath))

        assert restored.topk == original.topk
        assert restored.alpha == original.alpha
        assert restored.device == original.device
        assert restored.segmenter_model_name_or_path == original.segmenter_model_name_or_path

    def test_to_json_creates_formatted_output(self, tmp_path):
        """to_json should create formatted (indented) JSON."""
        config = HashformersConfig()
        filepath = tmp_path / "config.json"
        config.to_json(str(filepath))

        with open(filepath) as f:
            content = f.read()
        # Check for indentation (formatted JSON)
        assert '\n' in content
        assert '  ' in content  # indent=2


class TestHashformersConfigEdgeCases:
    """Test edge cases for HashformersConfig."""

    def test_zero_topk(self):
        """topk of 0 should be allowed."""
        config = HashformersConfig(topk=0)
        assert config.topk == 0

    def test_negative_alpha(self):
        """Negative alpha should be allowed."""
        config = HashformersConfig(alpha=-0.5)
        assert config.alpha == -0.5

    def test_float_precision(self):
        """Float values should maintain precision."""
        config = HashformersConfig(alpha=0.123456789)
        assert config.alpha == 0.123456789

    def test_empty_string_model(self):
        """Empty string model path should be allowed."""
        config = HashformersConfig(segmenter_model_name_or_path='')
        assert config.segmenter_model_name_or_path == ''

    def test_path_like_model(self):
        """Path-like model path should be allowed."""
        config = HashformersConfig(
            segmenter_model_name_or_path='/path/to/model'
        )
        assert config.segmenter_model_name_or_path == '/path/to/model'


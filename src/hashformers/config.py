"""
Configuration module for Hashformers.

This module provides centralized configuration management for hashformers,
enabling reproducibility and easy hyperparameter tuning.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class HashformersConfig:
    """Configuration for Hashformers segmentation.
    
    Attributes:
        topk: Number of top candidates to retain per step.
            Default: 20 (empirically determined for good coverage)
        steps: Maximum segmentation depth / number of beamsearch iterations.
            Default: 13 (sufficient for most hashtag lengths)
        alpha: Beamsearch score weight in ensemble.
            Default: 0.222 (empirically tuned on benchmark datasets)
        beta: Reranker score weight in ensemble.
            Default: 0.111 (empirically tuned on benchmark datasets)
        gpu_batch_size: Batch size for GPU inference.
            Default: 1000 (balance between speed and memory)
        device: Compute device ('cuda' or 'cpu').
            Default: 'cuda'
        segmenter_model_name_or_path: Path or name of the segmenter model.
            Default: 'gpt2'
        segmenter_model_type: Type of segmenter model.
            Default: 'gpt2'
        reranker_model_name_or_path: Path or name of the reranker model.
            Default: None (no reranker)
        reranker_model_type: Type of reranker model.
            Default: 'bert'
    """
    # Beamsearch hyperparameters
    topk: int = 20
    steps: int = 13
    
    # Ensemble weights (empirically tuned)
    alpha: float = 0.222
    beta: float = 0.111
    
    # GPU settings
    gpu_batch_size: int = 1000
    device: str = 'cuda'
    
    # Model settings
    segmenter_model_name_or_path: str = 'gpt2'
    segmenter_model_type: str = 'gpt2'
    reranker_model_name_or_path: Optional[str] = None
    reranker_model_type: str = 'bert'
    reranker_gpu_batch_size: int = 1000
    
    @classmethod
    def from_json(cls, path: str) -> 'HashformersConfig':
        """Load configuration from a JSON file.
        
        Args:
            path: Path to the JSON configuration file.
            
        Returns:
            HashformersConfig instance with loaded values.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            path: Path to save the JSON configuration file.
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HashformersConfig':
        """Create configuration from dictionary.
        
        Args:
            data: Dictionary with configuration values.
            
        Returns:
            HashformersConfig instance.
        """
        return cls(**data)


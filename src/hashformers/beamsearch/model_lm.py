"""Language Model wrapper with registry pattern for extensibility.

This module provides a unified interface for different language model types
(GPT-2, BERT, Seq2Seq) using native HuggingFace implementations.
"""

from typing import Dict, Type, Optional
import logging

from hashformers.beamsearch.gpt2_lm import GPT2LM
from hashformers.beamsearch.bert_lm import BertLM
from hashformers.beamsearch.seq2seq_lm import Seq2SeqLM

logger = logging.getLogger(__name__)


class ModelLM:
    """A Language Model (LM) class that supports GPT2, BERT, and Seq2Seq models.

    This class acts as a wrapper around the GPT2LM, BertLM, and Seq2SeqLM classes,
    providing a unified interface for interacting with any type of model. The specific
    type of model to use is determined by the 'model_type' argument provided during
    initialization.

    Uses a registry pattern for extensibility - new model types can be registered
    without modifying core code.

    Args:
        model_name_or_path (str, optional): The name or path of the pre-trained model.
        model_type (str, optional): The type of the model to use.
            - 'gpt2' or 'incremental': Auto-regressive models (GPT-2, GPT-J, etc.)
            - 'bert' or 'masked': Masked language models (BERT, RoBERTa, etc.)
            - 'seq2seq': Encoder-decoder models (FLAN-T5, T5, etc.)
        device (str, optional): The device on which to run computations. Defaults to 'cuda'.
        gpu_batch_size (int, optional): The batch size for GPU processing.
        gpu_id (int, optional): The ID of the GPU (kept for API compatibility).

    Raises:
        ValueError: If an unsupported 'model_type' is provided.

    Security Note:
        Models are loaded from Hugging Face Hub without signature verification.
        When loading models from untrusted sources, be aware that model files
        can contain arbitrary code. Consider using models only from trusted
        sources or implementing hash verification for production use.
    """

    # Registry for model types (HASH-005: Model Registry Pattern)
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class.

        Example:
            @ModelLM.register("custom-lm")
            class CustomLM:
                ...

        Args:
            name: The name to register the model under.

        Returns:
            Decorator function.
        """
        def decorator(model_cls: Type) -> Type:
            cls._registry[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available registered model types.

        Returns:
            List of registered model type names.
        """
        built_in = ['gpt2', 'incremental', 'bert', 'masked', 'seq2seq']
        return built_in + list(cls._registry.keys())

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model_type: Optional[str] = None,
        device: Optional[str] = 'cuda',
        gpu_batch_size: Optional[int] = None,
        gpu_id: int = 0
    ):
        self.gpu_batch_size = gpu_batch_size

        if model_type is None:
            self.model = None
            return

        # Log info for model loading
        if model_name_or_path:
            logger.debug(
                f"Loading model '{model_name_or_path}' of type '{model_type}'. "
                "Ensure you trust the source of this model."
            )

        # Check registry first for custom models
        if model_type in self._registry:
            model_cls = self._registry[model_type]
            self.model = model_cls(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size
            )
        # Built-in model types - GPT-2 / Incremental (Causal LM)
        elif model_type in ('gpt2', 'incremental'):
            self.model = GPT2LM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size or 20
            )
        # BERT / Masked LM
        elif model_type in ('bert', 'masked'):
            self.model = BertLM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size or 1,
                gpu_id=gpu_id
            )
        # Seq2Seq models (FLAN-T5, T5, etc.)
        elif model_type == 'seq2seq':
            self.model = Seq2SeqLM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size or 20
            )
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Available types: {self.get_available_models()}"
            )
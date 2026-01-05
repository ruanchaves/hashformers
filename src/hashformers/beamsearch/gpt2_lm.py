"""Native GPT-2 Language Model scorer using HuggingFace transformers.

This module provides a native implementation of GPT-2 scoring without
relying on the minicons library, enabling direct access to model internals
for optimizations like KV-caching.
"""

from typing import List, Optional, Tuple
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


class GPT2LM:
    """A Language Model (LM) scorer using GPT-2 or any causal LM.

    This class uses HuggingFace's AutoModelForCausalLM directly for scoring
    sentences, replacing the previous minicons-based implementation.

    Args:
        model_name_or_path (str): Name or path of the model to be used.
        device (str): The device to run the model on. Default is 'cuda'.
        gpu_batch_size (int): The batch size for GPU processing. Default is 20.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'cuda',
        gpu_batch_size: int = 20
    ):
        self.device = device
        self.gpu_batch_size = gpu_batch_size
        self.model_name_or_path = model_name_or_path

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()

        # Handle padding token - GPT-2 doesn't have one by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Set padding side to left for causal LM (standard practice)
        self.tokenizer.padding_side = 'left'

    def get_probs(self, list_of_candidates: List[str]) -> List[float]:
        """Compute probability scores for a list of candidate strings.

        Args:
            list_of_candidates: List of text strings to score.

        Returns:
            List of probability scores (lower is better, matching minicons convention).
        """
        probs = []
        dl = DataLoader(list_of_candidates, batch_size=self.gpu_batch_size)

        with torch.no_grad():
            for batch in dl:
                batch_probs = self._compute_batch_scores(list(batch))
                probs.extend(batch_probs)

        return probs

    def _compute_batch_scores(self, batch: List[str]) -> List[float]:
        """Compute scores for a batch of strings.

        Uses log-likelihood computation matching the minicons implementation.
        Returns 1 - sum(log_probs) so that lower scores are better.

        Args:
            batch: List of text strings to score.

        Returns:
            List of scores for each string in the batch.
        """
        # Tokenize with padding
        inputs = self.tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Compute per-token log probabilities
        # Shift logits and labels for causal LM (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['input_ids'][:, 1:].contiguous()
        shift_attention = inputs['attention_mask'][:, 1:].contiguous()

        # Get log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)

        # Gather the log probs for the actual tokens
        batch_size, seq_len = shift_labels.shape
        gathered_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding tokens
        gathered_log_probs = gathered_log_probs * shift_attention

        # Sum log probs for each sequence
        sum_log_probs = gathered_log_probs.sum(dim=-1)

        # Convert to scores (1 - sum so lower is better, matching minicons)
        scores = (1 - sum_log_probs).tolist()

        return scores

    def get_probs_with_cache(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[List[float], Tuple]:
        """Compute scores with KV-cache support for incremental inference.

        This method enables efficient incremental scoring by reusing
        previously computed key-value attention states.

        Args:
            input_ids: Token IDs to process (only new tokens if using cache).
            attention_mask: Optional attention mask.
            past_key_values: Cached key-value states from previous computation.

        Returns:
            Tuple of (scores, new_past_key_values) for continued caching.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device) if attention_mask is not None else None,
                past_key_values=past_key_values,
                use_cache=True
            )

        logits = outputs.logits
        new_past_key_values = outputs.past_key_values

        # Compute log probabilities for the last token(s)
        log_probs = torch.log_softmax(logits, dim=-1)

        # For incremental scoring, we typically want the prob of predicting
        # the next actual token, but this depends on the use case
        # Return the full log_probs and let the caller handle it

        return log_probs, new_past_key_values
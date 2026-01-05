"""Native BERT Language Model scorer using HuggingFace transformers.

This module provides a native implementation of BERT Pseudo-Log-Likelihood (PLL)
scoring without relying on the minicons library.
"""

from typing import List, Optional
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


class BertLM:
    """Implements a BERT-based language model scorer using Pseudo-Log-Likelihood.

    This class uses a transformer-based Masked Language Model (MLM) for scoring
    sentences. It computes the Pseudo-Log-Likelihood (PLL) by iteratively
    masking each token and computing the probability of the original token.

    Args:
        model_name_or_path (str): Identifier for the model to be loaded.
        device (str): The device to run the model on. Default is 'cuda'.
        gpu_batch_size (int): The batch size for GPU processing. Default is 1.
        gpu_id (int): Identifier of the GPU device (unused, kept for API compatibility).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'cuda',
        gpu_batch_size: int = 1,
        gpu_id: int = 0
    ):
        self.device = device
        self.gpu_batch_size = gpu_batch_size
        self.model_name_or_path = model_name_or_path

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()

        # Get special token IDs
        self.mask_token_id = self.tokenizer.mask_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        # Maximum sequence length for batched PLL
        self.max_batch_pll_length = 50  # Fall back to sequential for longer sequences

    def get_probs(self, list_of_candidates: List[str]) -> List[float]:
        """Compute Pseudo-Log-Likelihood scores for a list of candidate strings.

        Args:
            list_of_candidates: List of text strings to score.

        Returns:
            List of PLL scores (lower is better, matching minicons convention).
        """
        probs = []
        for candidate in list_of_candidates:
            pll = self._compute_pll(candidate)
            probs.append(pll)
        return probs

    def _compute_pll(self, text: str) -> float:
        """Compute Pseudo-Log-Likelihood for a single text.

        For a sentence with tokens [w1, w2, ..., wn]:
        PLL = sum over i of log P(wi | w1, ..., w_{i-1}, [MASK], w_{i+1}, ..., wn)

        Uses batched computation when sequence is short enough.

        Args:
            text: The text to score.

        Returns:
            Negative sum of log probabilities (lower is better).
        """
        # Tokenize the text
        tokens = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
        input_ids = tokens['input_ids'][0]  # Shape: (seq_len,)

        # Get the indices of tokens to mask (exclude [CLS] and [SEP])
        seq_len = len(input_ids)
        if seq_len <= 2:  # Only [CLS] and [SEP]
            return 0.0

        # Indices to mask: 1 to seq_len-2 (excluding [CLS] at 0 and [SEP] at -1)
        mask_indices = list(range(1, seq_len - 1))
        num_tokens = len(mask_indices)

        if num_tokens == 0:
            return 0.0

        # Use batched PLL for short sequences, sequential for long ones
        if num_tokens <= self.max_batch_pll_length:
            return self._compute_pll_batched(input_ids, mask_indices)
        else:
            return self._compute_pll_sequential(input_ids, mask_indices)

    def _compute_pll_batched(
        self,
        input_ids: torch.Tensor,
        mask_indices: List[int]
    ) -> float:
        """Compute PLL using batched inference (more efficient for short sequences).

        Creates all masked versions at once and runs them in a single forward pass.

        Args:
            input_ids: Original token IDs (1D tensor).
            mask_indices: Indices of tokens to mask.

        Returns:
            Negative sum of log probabilities.
        """
        num_tokens = len(mask_indices)

        # Create all masked versions at once
        # Shape: (num_tokens, seq_len)
        batched_ids = input_ids.unsqueeze(0).repeat(num_tokens, 1)

        # Store original tokens before masking
        original_tokens = []
        for i, mask_idx in enumerate(mask_indices):
            original_tokens.append(input_ids[mask_idx].item())
            batched_ids[i, mask_idx] = self.mask_token_id

        batched_ids = batched_ids.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(batched_ids)
            logits = outputs.logits  # Shape: (num_tokens, seq_len, vocab_size)

        # Compute log probabilities at each masked position
        log_prob_sum = 0.0
        for i, mask_idx in enumerate(mask_indices):
            token_logits = logits[i, mask_idx]  # Shape: (vocab_size,)
            log_probs = torch.log_softmax(token_logits, dim=-1)
            log_prob_sum += log_probs[original_tokens[i]].item()

        # Return negative sum (lower is better, matching minicons)
        return -log_prob_sum

    def _compute_pll_sequential(
        self,
        input_ids: torch.Tensor,
        mask_indices: List[int]
    ) -> float:
        """Compute PLL using sequential inference (for long sequences).

        Processes one masked position at a time to avoid OOM.

        Args:
            input_ids: Original token IDs (1D tensor).
            mask_indices: Indices of tokens to mask.

        Returns:
            Negative sum of log probabilities.
        """
        log_prob_sum = 0.0

        for mask_idx in mask_indices:
            # Create masked version
            masked_ids = input_ids.clone()
            original_token = masked_ids[mask_idx].item()
            masked_ids[mask_idx] = self.mask_token_id
            masked_ids = masked_ids.unsqueeze(0).to(self.device)

            # Forward pass
            with torch.no_grad():
                outputs = self.model(masked_ids)
                logits = outputs.logits[0, mask_idx]

            # Compute log probability
            log_probs = torch.log_softmax(logits, dim=-1)
            log_prob_sum += log_probs[original_token].item()

        # Return negative sum (lower is better, matching minicons)
        return -log_prob_sum
"""Native Seq2Seq Language Model scorer using HuggingFace transformers.

This module provides a native implementation of Seq2Seq model scoring
for models like FLAN-T5, without relying on the minicons library.
"""

from typing import List, Optional
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Seq2SeqLM:
    """A Language Model scorer for Seq2Seq models like FLAN-T5.

    This class uses HuggingFace's AutoModelForSeq2SeqLM for scoring
    sentences, typically used as a reranker.

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
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()

    def get_probs(self, list_of_candidates: List[str]) -> List[float]:
        """Compute probability scores for a list of candidate strings.

        For Seq2Seq models, we compute the probability of generating the
        candidate text given an empty or minimal source input.

        Args:
            list_of_candidates: List of text strings to score.

        Returns:
            List of probability scores (lower is better).
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

        Uses the model's log-likelihood of generating each candidate
        from a blank/minimal source input.

        Args:
            batch: List of text strings to score.

        Returns:
            List of scores for each string in the batch.
        """
        # For seq2seq scoring with blank source (matching minicons 'blank' format)
        # We use empty string or a minimal prompt as source
        source_texts = [""] * len(batch)

        # Tokenize source (encoder input)
        encoder_inputs = self.tokenizer(
            source_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        encoder_inputs = {k: v.to(self.device) for k, v in encoder_inputs.items()}

        # Tokenize targets (decoder input/output)
        with self.tokenizer.as_target_tokenizer():
            decoder_inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
        decoder_inputs = {k: v.to(self.device) for k, v in decoder_inputs.items()}

        # Forward pass with labels for loss computation
        outputs = self.model(
            input_ids=encoder_inputs['input_ids'],
            attention_mask=encoder_inputs['attention_mask'],
            decoder_input_ids=decoder_inputs['input_ids'],
            decoder_attention_mask=decoder_inputs['attention_mask'],
            labels=decoder_inputs['input_ids']
        )

        logits = outputs.logits

        # Compute per-token log probabilities
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = decoder_inputs['input_ids'][:, 1:].contiguous()
        shift_attention = decoder_inputs['attention_mask'][:, 1:].contiguous()

        # Get log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)

        # Gather the log probs for the actual tokens
        gathered_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding tokens
        gathered_log_probs = gathered_log_probs * shift_attention

        # Sum log probs for each sequence
        sum_log_probs = gathered_log_probs.sum(dim=-1)

        # Convert to scores (negative so lower is better)
        scores = (-sum_log_probs).tolist()

        return scores

import warnings
from typing import Callable

from hashformers.scoring import (
    IncrementalScorer,
    MaskedScorer,
    Seq2SeqScorer,
    canonicalize_model_type,
)

class MiniconsLM(object):

    scorer_map = {
        "incremental": IncrementalScorer,
        "masked": MaskedScorer,
        "seq2seq": Seq2SeqScorer,
    }

    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20, model_type='incremental'):
        self.model_type = canonicalize_model_type(model_type)
        scorer_cls = self.scorer_map[self.model_type]
        self.scorer = scorer_cls(
            model_name_or_path=model_name_or_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
        )
        self.gpu_batch_size = max(int(gpu_batch_size or 1), 1)
        self.model = self.scorer.model
        self.tokenizer = self.scorer.tokenizer
    
    def get_probs(self, list_of_candidates):
        probs = []
        for start in range(0, len(list_of_candidates), self.gpu_batch_size):
            batch = list_of_candidates[start : start + self.gpu_batch_size]
            probs.extend(self.get_batch_scores(batch))
        return probs
    
    def incremental_sequence_score(self, batch):
        return self.scorer.sequence_score(
            batch,
            reduction=lambda x: 1 - x.sum(0).item(),
            bos_token=True,
            eos_token=True,
        )

    def get_batch_scores(self, batch):
        if self.model_type == 'incremental':
            return self.incremental_sequence_score(batch)
        if self.model_type == 'masked':
            return self.scorer.sequence_score(batch, reduction=lambda x: x.sum(0).item())
        if self.model_type == 'seq2seq':
            return self.scorer.sequence_score(batch, source_format='blank')

        warnings.warn(
            f"Model type {self.model_type} is not explicitly handled; falling back to additive sequence scoring."
        )
        return self._fallback_sequence_score(batch, reduction=lambda x: x.sum(0).item())

    def _fallback_sequence_score(self, batch, reduction: Callable):
        return self.scorer.sequence_score(batch, reduction=reduction)

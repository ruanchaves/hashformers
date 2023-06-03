from minicons import scorer
from torch.utils.data import DataLoader
import warnings
import math

class MiniconsLM(object):

    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20, model_type='IncrementalLMScorer'):
        self.scorer = getattr(scorer, model_type)(model_name_or_path, device)
        self.gpu_batch_size = gpu_batch_size
        self.model_type = model_type
    
    def get_probs(self, list_of_candidates):
        probs = []
        dl = DataLoader(list_of_candidates, batch_size=self.gpu_batch_size)
        for batch in dl:
            probs.extend(self.get_batch_scores(batch))
        return probs
    
    def incremental_sequence_score(self, batch):
        tokens = self.scorer.prepare_text(batch, bos_token=True, eos_token=True)
        stats = self.scorer.compute_stats(tokens, prob=True)
        log_stats = [ [ math.log(x) for x in sequence ] for sequence in stats ]
        sum_log_stats = [ sum(x) for x in log_stats ]
        pos_sum_log_stats = [ 1 - x for x in sum_log_stats ]
        return pos_sum_log_stats

    def get_batch_scores(self, batch):
        if self.model_type == 'IncrementalLMScorer':
            return self.incremental_sequence_score(batch)
        elif self.model_type == 'MaskedLMScorer':
            return self.scorer.sequence_score(batch, reduction = lambda x: x.sum(0).item())
        elif self.model_type == 'Seq2SeqScorer':
            return self.scorer.sequence_score(batch, source_format = 'blank')
        else:
            warnings.warn(f"Model type {self.model_type} not implemented. Assuming reduction = lambda x: x.sum(0).item()")
            return self.scorer.sequence_score(batch, reduction = lambda x: x.sum(0).item())
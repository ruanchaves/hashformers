from minicons import scorer
from torch.utils.data import DataLoader
import warnings

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
    
    def get_batch_scores(self, batch):
        if self.model_type == 'IncrementalLMScorer':
            return self.scorer.sentence_score(batch, reduction = lambda x: -x.sum(0).item())
        elif self.model_type == 'MaskedLMScorer':
            return self.scorer.sentence_score(batch, reduction = lambda x: -x.sum(0).item())
        elif self.model_type == 'Seq2SeqScorer':
            return self.scorer.sentence_score(batch, source_format = 'blank')
        else:
            warnings.warn(f"Model type {self.model_type} not implemented. Assuming reduction = lambda x: -x.sum(0).item()")
            return self.scorer.sentence_score(batch, reduction = lambda x: -x.sum(0).item())
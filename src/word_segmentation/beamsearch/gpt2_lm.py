from lm_scorer.models.auto import GPT2LMScorer as LMScorer

class GPT2LM(object):

    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20):
        self.scorer = LMScorer(model_name_or_path, device=device, batch_size=gpu_batch_size)

    def get_probs(self, list_of_candidates):
        scores =  self.scorer.sentence_score(list_of_candidates, log=True)
        scores = [ 1-x for x in scores ]
        return scores
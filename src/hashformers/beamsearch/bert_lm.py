import mxnet as mx
import numpy as np
import pandas as pd
from mlm.models import get_pretrained
from mlm.scorers import MLMScorerPT

class BertLM(object):

    def __init__(self, model_name_or_path, gpu_batch_size=1, gpu_id=0):
        mx_device = [mx.gpu(gpu_id)]
        self.scorer = MLMScorerPT(*get_pretrained(mx_device, model_name_or_path), mx_device)
        self.gpu_batch_size = gpu_batch_size

    def get_probs(self, list_of_candidates):
        scores = []
        try:
            scores = self.scorer.score_sentences(list_of_candidates, split_size=self.gpu_batch_size)
            scores = [ x * -1 for x in scores ]
            return scores
        except:
            for candidate in list_of_candidates:
                try:
                    score = self.scorer.score_sentences([candidate])[0] if not pd.isna(candidate) else np.nan
                    score = score * -1
                except IndexError:
                    score = np.nan
                scores.append(score)
        return scores
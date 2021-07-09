import mx
import numpy as np
import pandas as pd
from mlm.models import get_pretrained
from mlm.scorers import MLMScorerPT

class BertLM(object):

    def __init__(self, model_name_or_path, *args, **kwargs):
        mx_device = [mx.gpu(0)]
        self.scorer = MLMScorerPT(*get_pretrained(mx_device, model_name_or_path), mx_device)

    def get_probs(self, list_of_candidates):
        scores = []
        for candidate in list_of_candidates:
            try:
                score = self.scorer.score_sentences([candidate])[0] if not pd.isna(candidate) else np.nan
                score = score * -1
            except IndexError:
                score = np.nan
            scores.append(score)
        return scores

def main():
    pass

if __name__ == '__main__':
    main()
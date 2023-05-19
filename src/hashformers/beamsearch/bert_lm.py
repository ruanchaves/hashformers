import mxnet as mx
import numpy as np
import pandas as pd
from mlm.models import get_pretrained
from mlm.scorers import MLMScorerPT

class BertLM(object):
    """
    Implements a BERT-based language model scorer, to compute sentence probabilities.
    This class uses a transformer-based Masked Language Model (MLM) for scoring. 
    
    Args:
        model_name_or_path (str): Identifier for the model to be loaded, which can be a model 
            name or the path to the directory where the model is stored.
        gpu_batch_size (int, optional): The size of the batch to be processed on the GPU. 
            Defaults to 1.
        gpu_id (int, optional): Identifier of the GPU device to be used. Defaults to 0.

    """
    def __init__(self, model_name_or_path, gpu_batch_size=1, gpu_id=0):
        mx_device = [mx.gpu(gpu_id)]
        self.scorer = MLMScorerPT(*get_pretrained(mx_device, model_name_or_path), mx_device)
        self.gpu_batch_size = gpu_batch_size

    def get_probs(self, list_of_candidates):
        """
        Returns probabilities for a list of candidate sentences.
        
        Args:
            list_of_candidates (list): A list of sentences for which the probability is to be 
                calculated. Each sentence should be a string.

        Returns:
            list: A list of probabilities corresponding to the input sentences. If an exception is encountered 
                while computing the probability for a sentence (e.g., if the sentence is not a string or 
                is NaN), the corresponding score in the output list is NaN.
        """
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
from hashformers.beamsearch.data_structures import (
    enforce_prob_dict,
    ProbabilityDictionary
)

class Reranker(object):

    def __init__(
        self,
        reranker
    ):
        self.model = reranker
    
    def rerank(
        self,
        data
    ):

        input_data = enforce_prob_dict(data)
        candidates = list(input_data.dictionary.keys())
        scores = self.model.get_probs(candidates)
        rank = { k:v for k,v in list(zip(candidates, scores))}
        return ProbabilityDictionary(rank)
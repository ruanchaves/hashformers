from hashformers.beamsearch.model_lm import ModelLM
from hashformers.beamsearch.data_structures import (
    enforce_prob_dict,
    ProbabilityDictionary
)

class Reranker(ModelLM):

    def __init__(
        self,
        model_name_or_path='bert-base-uncased',
        model_type='bert',
        gpu_batch_size=1,
        gpu_id=0
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_type=model_type,
            gpu_batch_size=gpu_batch_size,
            gpu_id=gpu_id
        )
    
    def rerank(
        self,
        data
    ):

        input_data = enforce_prob_dict(data)
        candidates = list(input_data.dictionary.keys())
        scores = self.model.get_probs(candidates)
        rank = { k:v for k,v in list(zip(candidates, scores))}
        return ProbabilityDictionary(rank)
from word_segmentation.beamsearch.model_lm import ModelLM
from word_segmentation.beamsearch.data_structures import ProbabilityDictionary

class Reranker(ModelLM):

    def __init__(
        self,
        model_name_or_path='bert-base-uncased',
        model_type='bert'
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_type=model_type,
        )
    
    def rerank(
        self,
        data
    ):
        if isinstance(data, ProbabilityDictionary):
            input_data = data
        elif isinstance(data, dict):
            input_data = ProbabilityDictionary(data)
        else:
            raise NotImplementedError
        
        top_2 = input_data.get_top_k(k=2)
        candidates = list(top_2.keys())
        scores = self.model.get_probs(candidates)
        rank = { k:v for k,v in list(zip(candidates, scores))}
        return rank
from hashformers.beamsearch.data_structures import (
    enforce_prob_dict,
    ProbabilityDictionary
)
from hashformers.beamsearch.model_lm import ModelLM

class Reranker(ModelLM):
    """
    A class that inherits from the ModelLM class and specializes in re-ranking given data.

    This class provides a method to re-rank data using a particular language model, which is determined
    during initialization from the parent ModelLM class. The re-ranking is performed based on the 
    probabilities produced by the language model.

    Args:
        model_name_or_path (str, optional): The name or path of the pre-trained model. Default is "bert-base-cased".
        model_type (str, optional): The type of the model to use. Default is "bert".
        gpu_batch_size (int, optional): The batch size to use when performing computations on the GPU. Default is 1000.
        gpu_id (int, optional): The ID of the GPU to use. Default is 0.
        device (str, optional): The device on which to run the computations. Default is "cuda".

    """
    def __init__(
        self,
        model_name_or_path="bert-base-cased",
        model_type="bert",
        gpu_batch_size=1000,
        gpu_id=0,
        device="cuda"
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            model_type=model_type,
            device=device,
            gpu_batch_size=gpu_batch_size,
            gpu_id=gpu_id
        )
    
    def rerank(
        self,
        data
    ):
        """
        Reranks the given data using the language model.

        The data is first converted to a ProbabilityDictionary, and then the model's probabilities 
        for each candidate in the data are computed. The results are returned as a new 
        ProbabilityDictionary.

        Args:
            data (dict or ProbabilityDictionary): The data to be reranked. If a dict is provided, it is 
                converted to a ProbabilityDictionary.

        Returns:
            ProbabilityDictionary: A ProbabilityDictionary of the reranked data, where the keys are 
                the candidates from the input data and the values are the scores computed by the 
                language model.
        """
        input_data = enforce_prob_dict(data)
        candidates = list(input_data.dictionary.keys())
        scores = self.model.get_probs(candidates)
        rank = { k:v for k,v in list(zip(candidates, scores))}
        return ProbabilityDictionary(rank)
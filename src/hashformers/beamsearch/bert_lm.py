from hashformers.beamsearch.minicons_lm import MiniconsLM

class BertLM(MiniconsLM):
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
        super().__init__(
            model_name_or_path=model_name_or_path,
            device='cuda',
            gpu_batch_size=gpu_batch_size,
            model_type='MaskedLMScorer'
        )
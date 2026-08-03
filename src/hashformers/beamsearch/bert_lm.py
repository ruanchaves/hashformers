from hashformers.beamsearch.minicons_lm import DEFAULT_MAX_BATCH_SIZE, MiniconsLM


class BertLM(MiniconsLM):
    """
    Implements a BERT-based language model scorer, to compute sentence probabilities.
    This class uses a transformer-based Masked Language Model (MLM) for scoring. 
    
    Args:
        model_name_or_path (str): Identifier for the model to be loaded, which can be a model 
            name or the path to the directory where the model is stored.
        gpu_batch_size (int or str, optional): A fixed batch size or ``"auto"``.
            Defaults to 1.
        max_gpu_batch_size (int, optional): Maximum automatic batch size.
            Defaults to 512.
        gpu_id (int, optional): Identifier of the GPU device to be used. Defaults to 0.
        device (str, optional): Device on which to run the model. Defaults to "cuda".

    """
    def __init__(
        self,
        model_name_or_path,
        gpu_batch_size=1,
        gpu_id=0,
        device='cuda',
        max_gpu_batch_size=DEFAULT_MAX_BATCH_SIZE,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
            model_type='MaskedLMScorer',
            max_gpu_batch_size=max_gpu_batch_size,
        )

from hashformers.beamsearch.minicons_lm import MiniconsLM

class GPT2LM(MiniconsLM):
    """A Language Model (LM) scorer using GPT2.

    This class utilizes the PaddedGPT2LMScorer for scoring sentences.

    Args:
        model_name_or_path (str): Name or path of the model to be used.
        device (str): The device to run the model on. Default is 'cuda'.
        gpu_batch_size (int): The batch size for GPU processing. Default is 20.
    """
    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
            model_type='IncrementalLMScorer'
        )
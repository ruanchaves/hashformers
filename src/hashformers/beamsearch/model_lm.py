from hashformers.beamsearch.bert_lm import BertLM
from hashformers.beamsearch.gpt2_lm import GPT2LM
from hashformers.beamsearch.minicons_lm import DEFAULT_MAX_BATCH_SIZE, MiniconsLM


class ModelLM(object):
    """
    A Language Model (LM) class that supports both GPT2 and BERT models.

    This class acts as a wrapper around the GPT2LM and BertLM classes, providing
    a unified interface for interacting with either type of model. The specific
    type of model to use is determined by the 'model_type' argument provided during
    initialization.

    Args:
        model_name_or_path (str, optional): The name or path of the pre-trained model.
        model_type (str, optional): The type of the model to use. Should be either 'gpt2' or 'bert'.
        device (str, optional): The device on which to run the computations. Defaults to None which implies CPU.
        gpu_batch_size (int or str, optional): A fixed batch size or ``"auto"``.
        max_gpu_batch_size (int, optional): Maximum automatic batch size.
        gpu_id (int, optional): The ID of the GPU to use. Only relevant if 'model_type' is 'bert'. Default is 0.

    Raises:
        ValueError: If an unsupported 'model_type' is provided.
    """
    def __init__(
        self,
        model_name_or_path=None,
        model_type=None,
        device=None,
        gpu_batch_size=None,
        gpu_id=0,
        max_gpu_batch_size=DEFAULT_MAX_BATCH_SIZE,
    ):
        self.gpu_batch_size = gpu_batch_size
        self.max_gpu_batch_size = max_gpu_batch_size
        if model_type is None:
            self.model = None
        elif model_type == 'gpt2':
            self.model = GPT2LM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size,
                max_gpu_batch_size=max_gpu_batch_size,
            )
        elif model_type == 'bert':
            self.model = BertLM(
                model_name_or_path,
                gpu_batch_size=gpu_batch_size,
                gpu_id=gpu_id,
                device=device,
                max_gpu_batch_size=max_gpu_batch_size,
            )
        elif model_type == 'seq2seq':
            self.model = MiniconsLM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size,
                model_type='Seq2SeqScorer',
                max_gpu_batch_size=max_gpu_batch_size,
            )
        elif model_type == 'masked':
            self.model = MiniconsLM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size,
                model_type='MaskedLMScorer',
                max_gpu_batch_size=max_gpu_batch_size,
            )
        elif model_type == 'incremental':
            self.model = MiniconsLM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size,
                model_type='IncrementalLMScorer',
                max_gpu_batch_size=max_gpu_batch_size,
            )
        else:
            self.model = MiniconsLM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size,
                model_type=model_type,
                max_gpu_batch_size=max_gpu_batch_size,
            )

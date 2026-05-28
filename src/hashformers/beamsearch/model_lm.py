from hashformers.beamsearch.minicons_lm import MiniconsLM

class ModelLM(object):
    """
    A Language Model (LM) wrapper that routes supported model types to
    hashformers' internal transformer scorers.

    Args:
        model_name_or_path (str, optional): The name or path of the pre-trained model.
        model_type (str, optional): The type of the model to use.
        device (str, optional): The device on which to run the computations. Defaults to None which implies CPU.
        gpu_batch_size (int, optional): The batch size to use when performing computations on the GPU.
        gpu_id (int, optional): Reserved for backward compatibility. Default is 0.

    Raises:
        ValueError: If an unsupported 'model_type' is provided.
    """
    def __init__(self, model_name_or_path=None, model_type=None, device=None, gpu_batch_size=None, gpu_id=0):
        self.gpu_batch_size = gpu_batch_size
        if model_type is None:
            self.model = None
        else:
            self.model = MiniconsLM(
                model_name_or_path,
                device=device,
                gpu_batch_size=gpu_batch_size,
                model_type=model_type,
            )

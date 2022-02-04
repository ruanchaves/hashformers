class ModelLM(object):

    def __init__(self, model_name_or_path=None, model_type=None, device=None, gpu_batch_size=None, gpu_id=0):
        self.gpu_batch_size = gpu_batch_size
        if model_type == 'gpt2':
            from hashformers.beamsearch.gpt2_lm import GPT2LM
            self.model = GPT2LM(model_name_or_path, device=device, gpu_batch_size=gpu_batch_size)
        elif model_type == 'bert':
            from hashformers.beamsearch.bert_lm import BertLM
            self.model = BertLM(model_name_or_path, gpu_batch_size=gpu_batch_size, gpu_id=gpu_id)
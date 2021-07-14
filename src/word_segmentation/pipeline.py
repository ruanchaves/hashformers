from word_segmentation.beamsearch.algorithm import Beamsearch
from word_segmentation.beamsearch.reranker import Reranker

class Pipeline(object):

    def __init__(
        self,
        decoder_model_name_or_path = "gpt2-large",
        decoder_model_type = "gpt2",
        decoder_device = "cuda",
        decoder_gpu_batch_size = 1,
        encoder_model_name_or_path = None,
        encoder_model_type = "bert"
    ):
        self.decoder_model = Beamsearch(
        model_name_or_path=decoder_model_name_or_path,
        model_type=decoder_model_type,
        device=decoder_device,
        gpu_batch_size=decoder_gpu_batch_size
    )

        if encoder_model_name_or_path:
            self.encoder_model = Reranker(
                model_name_or_path=encoder_model_name_or_path,
                model_type=encoder_model_type
            )
        else:
            self.encoder_model = None
    
    def segment(
            dataset_or_path, 
            evaluate=True,
            sample=None,
            topk=20,
            steps=13):
        if isinstance(dataset_or_path, str):
            dataset = load_dataset('text', data_files={'test': dataset_or_path})
            dataset = dataset['test'].to_dict()['text']
        elif isinstance(dataset_or_path, list) and \
            all(isinstance(x, str) for x in dataset_or_path):
            dataset = dataset_or_path
        else:
            raise NotImplementedError
        
        dataset = [ x.strip() for x in dataset ]
        if evaluate:
            characters = [
                x.replace(" ", "") for x in dataset
            ]
        else:
            characters = dataset
        
        if sample:
            characters = characters[0:sample]
            dataset = dataset[0:sample]
        
        decoder_run = self.decoder_model.run(
            characters,
            topk=topk,
            steps=steps
        )

        if self.encoder_model:
            encoder_run = self.encoder_model.run
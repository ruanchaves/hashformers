from word_segmentation.beamsearch.algorithm import Beamsearch
from word_segmentation.beamsearch.reranker import Reranker
from word_segmentation.beamsearch.data_structures import enforce_prob_dict
from word_segmentation.ensemble.top2_fusion import top2_ensemble

class WordSegmenter(object):

    def __init__(
        self,
        decoder_model_name_or_path = "gpt2",
        decoder_model_type = "gpt2",
        decoder_device = "cuda",
        decoder_gpu_batch_size = 1,
        encoder_model_name_or_path = "bert-base-uncased",
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
            word_list,
            topk=20,
            steps=13,
            alpha=0.222,
            beta=0.111):

        decoder_run = self.decoder_model.run(
            word_list,
            topk=topk,
            steps=steps
        )
        
        if self.encoder_model:
            encoder_run = self.encoder_model.rerank(decoder_run)

            ensemble = top2_ensemble(
                decoder_run,
                encoder_run,
                alpha=alpha,
                beta=beta
            )
        
        if self.encoder_model:
            ensemble_prob_dict = enforce_prob_dict(
                ensemble,
                score_field="ensemble_rank")
            segs = ensemble_prob_dict.get_segmentations(
                astype="list",
                gold_array=word_list
            )
        else:
            decoder_prob_dict = enforce_prob_dict(
                decoder_run,
                score_field="score"
            )
            segs = decoder_prob_dict.get_segmentation(
                astype="list",
                gold_array=word_list
            )

        return segs
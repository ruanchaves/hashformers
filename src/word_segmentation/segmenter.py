import itertools
from word_segmentation.beamsearch.algorithm import Beamsearch
from word_segmentation.beamsearch.reranker import Reranker
from word_segmentation.beamsearch.data_structures import enforce_prob_dict
from word_segmentation.ensemble.top2_fusion import top2_ensemble
from spacy.tokenizer import _get_regex_pattern
import spacy
import re 
import itertools
from collections import namedtuple

WordSegmenterResult = namedtuple(
    "WordSegmenterResult", 
    [
        "dataset", 
        "hashtag_dict"
    ]
)

class WordSegmenter(object):

    def __init__(
        self,
        segmenter_model_name_or_path = "gpt2",
        segmenter_model_type = "gpt2",
        segmenter_device = "cuda",
        segmenter_gpu_batch_size = 1,
        reranker_model_name_or_path = "bert-base-uncased",
        reranker_model_type = "bert"
    ):

        self.segmenter_model = Beamsearch(
        model_name_or_path=segmenter_model_name_or_path,
        model_type=segmenter_model_type,
        device=segmenter_device,
        gpu_batch_size=segmenter_gpu_batch_size
    )

        if reranker_model_name_or_path:
            self.reranker_model = Reranker(
                model_name_or_path=reranker_model_name_or_path,
                model_type=reranker_model_type
            )
        else:
            self.reranker_model = None

    def segment(
            self,
            word_list,
            topk=20,
            steps=13,
            alpha=0.222,
            beta=0.111,
            use_reranker=True,
            return_ranks=False):

        segmenter_run = self.segmenter_model.run(
            word_list,
            topk=topk,
            steps=steps
        )
        
        if self.reranker_model:
            reranker_run = self.reranker_model.rerank(segmenter_run)

            ensemble = top2_ensemble(
                segmenter_run,
                reranker_run,
                alpha=alpha,
                beta=beta
            )
        
        if self.reranker_model and use_reranker:
            ensemble_prob_dict = enforce_prob_dict(
                ensemble,
                score_field="ensemble_rank")
            segs = ensemble_prob_dict.get_segmentations(
                astype="list",
                gold_array=word_list
            )
        else:
            segmenter_prob_dict = enforce_prob_dict(
                segmenter_run,
                score_field="score"
            )
            segs = segmenter_prob_dict.get_segmentation(
                astype="list",
                gold_array=word_list
            )

        if not return_ranks:
            return segs
        else:
            segmenter_df = segmenter_run.to_dataframe()
            if use_reranker:
                reranker_df = reranker_run.to_dataframe()
            else:
                reranker_df = None
            return {
                "segmenter": segmenter_df,
                "reranker": reranker_df,
                "ensemble": ensemble,
                "segmentations": segs
            }
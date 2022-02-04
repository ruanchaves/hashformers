from hashformers.beamsearch.algorithm import Beamsearch
from hashformers.beamsearch.reranker import Reranker
from hashformers.beamsearch.data_structures import enforce_prob_dict
from hashformers.ensemble.top2_fusion import top2_ensemble
from typing import List, Union, Any

class WordSegmenter(object):

    def __init__(
        self,
        segmenter_model_name_or_path = "gpt2",
        segmenter_model_type = "gpt2",
        segmenter_device = "cuda",
        segmenter_gpu_batch_size = 1,
        reranker_gpu_batch_size = 2000,
        reranker_model_name_or_path = "bert-base-uncased",
        reranker_model_type = "bert"
    ):
        """Word segmentation API initialization. 
           A GPT-2 model must be passed to `segmenter_model_name_or_path`, and optionally a BERT model to `reranker_model_name_or_path`.
           If `reranker_model_name_or_path` is set to `False` or `None`, the word segmenter object will work without a reranker.


        Args:
            segmenter_model_name_or_path (str, optional): GPT-2 that will be fetched from the Hugging Face Model Hub. Defaults to "gpt2".
            segmenter_model_type (str, optional): Transformer decoder model type. Defaults to "gpt2".
            segmenter_device (str, optional): Device. Defaults to "cuda".
            segmenter_gpu_batch_size (int, optional): Segmenter GPU batch size. Defaults to 1.
            reranker_gpu_batch_size (int, optional): Reranker GPU split size. Defaults to 2000.
            reranker_model_name_or_path (str, optional): BERT model that will be fetched from the Hugging Face Model Hub. It is possible to turn off the reranker by passing a None or False value to this argument. Defaults to "bert-base-uncased".
            reranker_model_type (str, optional): Transformer encoder model type. Defaults to "bert".
        """
        self.segmenter_model = Beamsearch(
        model_name_or_path=segmenter_model_name_or_path,
        model_type=segmenter_model_type,
        device=segmenter_device,
        gpu_batch_size=segmenter_gpu_batch_size
    )

        if reranker_model_name_or_path:
            self.reranker_model = Reranker(
                model_name_or_path=reranker_model_name_or_path,
                model_type=reranker_model_type,
                gpu_batch_size=reranker_gpu_batch_size
            )
        else:
            self.reranker_model = None

    def segment(
            self,
            word_list: List[str],
            topk: int = 20,
            steps: int = 13,
            alpha: float = 0.222,
            beta: float = 0.111,
            use_reranker: bool = True,
            return_ranks: bool = False,
            trim_hashtags: bool = True) -> Any :
        """Segment a list of hashtags.

        Args:
            word_list (List[str]): A list of hashtag strings.
            topk (int, optional): 
                top-k parameter for the Beamsearch algorithm. 
                A lower top-k value will speed up the algorithm. 
                However, this will decrease the amount of candidate segmentations in a rank. 
                Defaults to 20.
            steps (int, optional): 
                steps parameter for the Beamsearch algorithm. 
                A lower amount of steps will speed up the algorithm. 
                However, the algorithm will never detect a number of words larger than amount of steps. 
                Defaults to 13. 
            alpha (float, optional): 
                alpha parameter for the top-2 ensemble. 
                It controls the weight given to the segmenter candidates. 
                Reasonable values range from 0 to 1. 
                Defaults to 0.222.
            beta (float, optional): 
                beta parameter for the top-2 ensemble. 
                It controls the weight given to the reranker candidates. 
                Reasonable values range from 0 to 1. 
                Defaults to 0.111.
            use_reranker (bool, optional): 
                Whether or not to run the reranker. 
                Defaults to True.
            return_ranks (bool, optional): 
                Return not just the segmented hashtags but also the a dictionary of the ranks. 
                Defaults to False.
            trim_hashtags (bool, optional): 
                Automatically remove "#" characters from the beginning of the hashtags. 
                Defaults to True.

        Returns:
            Any: A list of segmented hashtags if return_ranks == False. A dictionary of the ranks and the segmented hashtags if return_ranks == True.
        """

        if trim_hashtags:
            word_list = \
                [ x.lstrip("#") for x in word_list ]

        segmenter_run = self.segmenter_model.run(
            word_list,
            topk=topk,
            steps=steps
        )
        
        ensemble = None
        if use_reranker:
            reranker_run = self.reranker_model.rerank(segmenter_run)

            ensemble = top2_ensemble(
                segmenter_run,
                reranker_run,
                alpha=alpha,
                beta=beta
            )

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
            segs = segmenter_prob_dict.get_segmentations(
                astype="list",
                gold_array=word_list
            )

        if not return_ranks:
            return segs
        else:
            segmenter_df = segmenter_run.to_dataframe().reset_index(drop=True)
            if use_reranker:
                reranker_df = reranker_run.to_dataframe().reset_index(drop=True)
            else:
                reranker_df = None
            return {
                "segmenter": segmenter_df,
                "reranker": reranker_df,
                "ensemble": ensemble,
                "segmentations": segs
            }
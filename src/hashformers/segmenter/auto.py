from hashformers.segmenter import (
    BaseWordSegmenter
)

from hashformers.beamsearch.algorithm import Beamsearch
from hashformers.beamsearch.reranker import Reranker
from hashformers.ensemble.top2_fusion import Top2_Ensembler

class TransformerWordSegmenter(BaseWordSegmenter):
    def __init__(
        self,
        segmenter_model_name_or_path = "gpt2",
        segmenter_model_type = "incremental",
        segmenter_device = "cuda",
        segmenter_gpu_batch_size = 1000,
        reranker_gpu_batch_size = 1000,
        reranker_model_name_or_path = None,
        reranker_model_type = "masked",
        reranker_device = "cuda"
    ):
        """Word segmentation API initialization. 
           A causal language model must be passed to `segmenter_model_name_or_path`, and optionally
           a masked or seq2seq language model to `reranker_model_name_or_path`.
           If `reranker_model_name_or_path` is set to `False` or `None`, the word segmenter object will work without a reranker.


        Args:
            segmenter_model_name_or_path (str, optional): Causal language model that will be fetched from the Hugging Face Model Hub. Defaults to "gpt2".
            segmenter_model_type (str, optional): Transformer decoder model type. Defaults to "incremental".
            segmenter_device (str, optional): Device. Defaults to "cuda".
            segmenter_gpu_batch_size (int, optional): Segmenter GPU batch size. Defaults to 1.
            reranker_gpu_batch_size (int, optional): Reranker GPU split size. Defaults to 2000.
            reranker_model_name_or_path (str, optional): Masked or seq2seq model that will be fetched from the Hugging Face Model Hub. It is possible to turn off the reranker by passing a None or False value to this argument. Defaults to None.
            reranker_model_type (str, optional): Transformer reranker model type. Defaults to "masked".
        """
        segmenter_model = Beamsearch(
        model_name_or_path=segmenter_model_name_or_path,
        model_type=segmenter_model_type,
        device=segmenter_device,
        gpu_batch_size=segmenter_gpu_batch_size
    )

        if reranker_model_name_or_path:
            reranker_model = Reranker(
                model_name_or_path=reranker_model_name_or_path,
                model_type=reranker_model_type,
                gpu_batch_size=reranker_gpu_batch_size,
                device=reranker_device
            )
        else:
            reranker_model = None

        ensembler = Top2_Ensembler()

        super().__init__(
            segmenter=segmenter_model,
            reranker=reranker_model,
            ensembler=ensembler
        )

    def segment(
            self,
            word_list,
            topk: int = 20,
            steps: int = 13,
            alpha: float = 0.222,
            beta: float = 0.111,
            use_reranker: bool = True,
            return_ranks: bool = False):

            segmenter_kwargs = {
                "topk": topk,
                "steps": steps
            }

            ensembler_kwargs = {
                "alpha": alpha,
                "beta": beta
            }

            return super().segment(
                word_list,
                segmenter_kwargs=segmenter_kwargs,
                ensembler_kwargs=ensembler_kwargs,
                use_reranker=use_reranker,
                return_ranks=return_ranks
            )

from hashformers.beamsearch.data_structures import enforce_prob_dict
from typing import Any
from hashformers.segmenter.base_segmenter import BaseSegmenter
from hashformers.segmenter.data_structures import WordSegmenterOutput


class BaseWordSegmenter(BaseSegmenter):
    """
    Initializes BaseWordSegmenter class with segmenter, reranker and ensembler models.

    Args:
        segmenter: The model used for initial word segmentation.
        reranker: The model used for reranking the segmented words.
        ensembler: The model used for ensemble operations over the segmenter and reranker models.
    """
    def __init__(
        self,
        segmenter = None,
        reranker = None,
        ensembler = None
    ):
        self.segmenter_model = segmenter
        self.reranker_model = reranker
        self.ensembler = ensembler

    def get_segmenter(self):
        """
        Returns the segmenter model.
        """
        return self.segmenter_model.model

    def get_reranker(self):
        """
        Returns the reranker model.
        """
        return self.reranker_model.model

    def get_ensembler(self):
        """
        Returns the ensembler model.
        """
        return self.ensembler

    def set_segmenter(self, segmenter):
        """
        Sets the segmenter model.

        Args:
            segmenter: The model used for initial hashtag segmentation.
        """
        self.segmenter_model.model = segmenter
    
    def set_reranker(self, reranker):
        """
        Sets the reranker model.

        Args:
            reranker: The model used for reranking the segmented hashtags.
        """
        self.reranker_model.model = reranker

    def set_ensembler(self, ensembler):
        """
        Sets the ensembler model.

        Args:
            ensembler: The model used for ensemble operations over the segmenter and reranker models.
        """
        self.ensembler = ensembler

    def segment(
            self,
            word_list: list[str],
            segmenter_run: Any = None,
            preprocessing_kwargs: dict = {},
            segmenter_kwargs: dict = {},
            ensembler_kwargs: dict = {},
            reranker_kwargs: dict = {},
            use_reranker: bool = True,
            use_ensembler: bool = True,
            return_ranks: bool = False) -> Any :
        """
        Segments the input list of words using the segmenter, reranker, and ensembler models.
        Allows customization of the segmenting process with multiple keyword arguments.

        Args:
            word_list: List of strings, where each string is a word to be segmented.
            segmenter_run: Optional argument to use a pre-existing segmenter run, defaults to None.
            preprocessing_kwargs: Keyword arguments to be used during the preprocessing phase.
            segmenter_kwargs: Keyword arguments to be used by the segmenter model.
            ensembler_kwargs: Keyword arguments to be used by the ensembler model.
            reranker_kwargs: Keyword arguments to be used by the reranker model.
            use_reranker: Boolean flag to indicate whether to use the reranker model, defaults to True.
            use_ensembler: Boolean flag to indicate whether to use the ensembler model, defaults to True.
            return_ranks: Boolean flag to indicate whether to return the ranks from the models, defaults to False.

        Returns:
            Returns the segmented words. If return_ranks is True, also returns the segmenter_rank, reranker_rank, and ensemble_rank.
        """
        word_list = super().preprocess(word_list, **preprocessing_kwargs)

        if segmenter_run is None:
            segmenter_run = self.segmenter_model.run(
                word_list,
                **segmenter_kwargs
            )
        
        ensemble_prob_dict = None

        segmenter_prob_dict = enforce_prob_dict(
                segmenter_run,
                score_field="score"
        )

        if use_reranker and self.reranker_model:
            reranker_run = self.reranker_model.rerank(segmenter_run, **reranker_kwargs)

        if use_reranker and self.reranker_model and use_ensembler and self.ensembler:
            ensemble_prob_dict = self.ensembler.run(
                segmenter_run,
                reranker_run,
                **ensembler_kwargs
            )
            segs = ensemble_prob_dict.get_segmentations(
                astype="list",
                gold_array=word_list
            )

        elif use_reranker and self.reranker_model:
            segs = reranker_run.get_segmentations(
                astype="list",
                gold_array=word_list
            )

        else:
            segs = segmenter_prob_dict.get_segmentations(
                astype="list",
                gold_array=word_list
            )

        if not return_ranks:
            return segs
        else:
            segmenter_df = segmenter_prob_dict.to_dataframe().reset_index(drop=True)
            reranker_df = None
            ensembler_df = None

            if use_reranker:
                if self.reranker_model:
                    reranker_df = reranker_run.to_dataframe().reset_index(drop=True)
                if use_ensembler and self.ensembler and ensemble_prob_dict: 
                    ensembler_df = ensemble_prob_dict.to_dataframe().reset_index(drop=True)

            return WordSegmenterOutput(
                segmenter_rank=segmenter_df,
                reranker_rank=reranker_df,
                ensemble_rank=ensembler_df,
                output=segs
            )

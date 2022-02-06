import hashformers
from hashformers.beamsearch.algorithm import Beamsearch
from hashformers.beamsearch.reranker import Reranker
from hashformers.beamsearch.data_structures import enforce_prob_dict
from hashformers.ensemble.top2_fusion import top2_ensemble
from typing import List, Union, Any
from dataclasses import dataclass
import pandas as pd
from ttp import ttp
from ekphrasis.classes.segmenter import Segmenter as EkphrasisSegmenter
import re
import typing
import inspect
import copy
import torch
from math import log10

@dataclass
class WordSegmenterOutput:
    output: List[str]
    segmenter_rank: Union[pd.DataFrame, None] = None
    reranker_rank: Union[pd.DataFrame, None] = None
    ensemble_rank: Union[pd.DataFrame, None] = None

@dataclass
class TweetSegmenterOutput:
    output: List[str]
    word_segmenter_output: Any

def prune_segmenter_layers(ws, layer_list=[0]):
    ws.segmenter_model.model.scorer.model = \
        deleteEncodingLayers(ws.segmenter_model.model.scorer.model, layer_list=layer_list)
    return ws

def deleteEncodingLayers(model, layer_list=[0]):
    oldModuleList = model.transformer.h
    newModuleList = torch.nn.ModuleList()

    for index in layer_list:
        newModuleList.append(oldModuleList[index])

    copyOfModel = copy.deepcopy(model)
    copyOfModel.transformer.h = newModuleList

    return copyOfModel

class BaseSegmenter(object):

    def predict(self, input, *args, **kwargs):
        first_argument = inspect.getfullargspec(self.segment).args[1]
        first_argument_type = typing.get_type_hints(self.segment)[first_argument]
        a = type(first_argument_type) == type(str)
        b = type(input) == type(str)
        output = None
        if a and b:
            output = self.segment(input, *args, **kwargs)
        elif not a and not b:
            output = self.segment(input, *args, **kwargs)
        elif a and not b:
            output = [ self.segment(x, *args, **kwargs) for x in input ]
        elif not a and b:
            output = self.segment([input], *args, **kwargs)[0]

        if type(output) == type(WordSegmenterOutput):
            return output

        if type(output) == type(TweetSegmenterOutput):
            return output

        if type(output) != type(WordSegmenterOutput):
            output = WordSegmenterOutput(output=output)
        
        return output

class EkphrasisWordSegmenter(EkphrasisSegmenter, BaseSegmenter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def find_segment(self, text, prev='<S>'):
        if not text:
            return 0.0, []
        candidates = [self.combine((log10(self.condProbWord(first, prev)), first), self.find_segment(rem, first))
                      for first, rem in self.splits(text)]
        return max(candidates)

    def segment(self, word: str) -> str:
        if word.islower():
            return " ".join(self.find_segment(word)[1])
        else:
            return self.case_split.sub(r' \1', word).lower()

class RegexWordSegmenter(BaseSegmenter):

    def __init__(self,regex_rules=None):
        if not regex_rules:
            regex_rules = [r'([A-Z]+)']
        self.regex_rules = [
            re.compile(x) for x in regex_rules
        ]

    def segment_word(self, rule, word):
        return rule.sub(r' \1', word).strip()

    def segment(self, word_list: List[str]):
        for rule in self.regex_rules:
            for idx, word in enumerate(word_list):
                word_list[idx] = self.segment_word(rule, word)
        return word_list

class WordSegmenter(BaseSegmenter):
    """A general-purpose word segmentation API.
    """
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
            return_ranks: bool = False) -> Any :
        """Segment a list of strings.

        Args:
            word_list (List[str]): A list of strings.
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

        Returns:
            Any: A list of segmented words if return_ranks == False. A dictionary of the ranks and the segmented words if return_ranks == True.
        """

        segmenter_run = self.segmenter_model.run(
            word_list,
            topk=topk,
            steps=steps
        )
        
        ensemble = None
        if use_reranker and self.reranker_model:
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
            return WordSegmenterOutput(
                segmenter_rank=segmenter_df,
                reranker_rank=reranker_df,
                ensemble_rank=ensemble,
                output=segs
            )

class TwitterTextMatcher(object):
    
    def __init__(self):
        self.parser = ttp.Parser()
    
    def __call__(self, tweets):
        return [ self.parser.parse(x).tags for x in tweets ]
    
class TweetSegmenter(BaseSegmenter):

    def __init__(self, matcher=None, word_segmenter=None):

        if matcher:
            self.matcher = matcher
        else:
            self.matcher = TwitterTextMatcher()

        if word_segmenter:
            self.word_segmenter = word_segmenter
        else:
            self.word_segmenter = RegexWordSegmenter()

    def extract_hashtags(self, tweets):
        return self.matcher(tweets)

    def create_regex_pattern(self, replacement_dict, flags=0):
        return re.compile("|".join(replacement_dict), flags)

    def compile_dict(self, hashtags, segmentations, hashtag_token=None, lower=False, separator=" ", hashtag_character="#"):

        hashtag_buffer = {
            k:v for k,v in zip(hashtags, segmentations)
        }

        replacement_dict = {}

        for key, value in hashtag_buffer.items():

            if not key.startswith(hashtag_character):
                hashtag_key = hashtag_character + key
            else:
                hashtag_key = key

            if hashtag_token:
                hashtag_value = hashtag_token + separator + value
            else:
                hashtag_value = value

            if lower:
                hashtag_value = hashtag_value.lower()

            replacement_dict.update(hashtag_key, hashtag_value)

        # Treat edge case: overlapping hashtags
        replacement_dict = \
            map(re.escape, sorted(replacement_dict, key=len, reverse=True))

        return replacement_dict

    def replace_hashtags(self, tweet, regex_pattern, replacement_dict):

        if not replacement_dict:
            return tweet

        tweet = regex_pattern.sub(lambda m: replacement_dict[m.group(0)], tweet)
        
        return tweet

    def segment(self, tweets: str, regex_flag: Any = 0, preprocessing_kwargs: dict = {}, segmenter_kwargs: dict = {} ):

        hashtags = self.extract_hashtags(tweets)
        
        word_segmenter_output = self.word_segmenter.predict(hashtags, **segmenter_kwargs)

        segmentations = word_segmenter_output.output

        replacement_dict = self.compile_dict(hashtags, segmentations, **preprocessing_kwargs)

        regex_pattern = self.create_regex_pattern(replacement_dict, flag=regex_flag)

        tweets = [ self.replace_hashtags(tweet, regex_pattern, replacement_dict) for tweet in tweets]
        
        return TweetSegmenterOutput(
            word_segmenter_output = word_segmenter_output,
            output = tweets
        )
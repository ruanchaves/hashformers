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
import copy
import torch
from math import log10
from functools import reduce
import dataclasses
from collections.abc import Iterable

@dataclass
class WordSegmenterOutput:
    output: List[str]
    segmenter_rank: Union[pd.DataFrame, None] = None
    reranker_rank: Union[pd.DataFrame, None] = None
    ensemble_rank: Union[pd.DataFrame, None] = None

@dataclass
class HashtagContainer:
    hashtags: List[List[str]]
    hashtag_set: List[str]
    replacement_dict: dict

@dataclass
class TweetSegmenterOutput:
    output: List[str]
    word_segmenter_output: Any

def coerce_segmenter_objects(method):
    def wrapper(inputs, *args, **kwargs):
        if isinstance(inputs, str):
            output = method([inputs], *args, **kwargs)
        else:
            output = method(inputs, *args, **kwargs)
        
        for allowed_type in [
            WordSegmenterOutput,
            TweetSegmenterOutput
        ]:
            if isinstance(output, allowed_type):
                return output
        
        if isinstance(output, str):
            return WordSegmenterOutput(output=[output])

        if isinstance(output, Iterable):
            return WordSegmenterOutput(output=output)

    return wrapper

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

    @coerce_segmenter_objects
    def predict(self, *args, **kwargs):
        return self.segment(*args, **kwargs)

class EkphrasisWordSegmenter(EkphrasisSegmenter, BaseSegmenter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def find_segment(self, text, prev='<S>'):
        if not text:
            return 0.0, []
        candidates = [self.combine((log10(self.condProbWord(first, prev)), first), self.find_segment(rem, first))
                      for first, rem in self.splits(text)]
        return max(candidates)

    def segment_word(self, word) -> str:
        if word.islower():
            return " ".join(self.find_segment(word)[1])
        else:
            return self.case_split.sub(r' \1', word).lower()

    def segment(self, inputs) -> List[str]:
        return [ self.segment_word(word) for word in inputs ]

class RegexWordSegmenter(BaseSegmenter):

    def __init__(self,regex_rules=None):
        if not regex_rules:
            regex_rules = [r'([A-Z]+)']
        self.regex_rules = [
            re.compile(x) for x in regex_rules
        ]

    def segment_word(self, rule, word):
        return rule.sub(r' \1', word).strip()

    def segmentation_generator(self, word_list):
        for rule in self.regex_rules:
            for idx, word in enumerate(word_list):
                yield self.segment_word(rule, word)

    def segment(self, inputs: List[str]):
        return list(self.segmentation_generator(inputs))

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
            use_reranker: bool = False,
            return_ranks: bool = False) -> Any :
        """Segment a list of strings.

        :param word_list: A list of strings.
        :type word_list: List[str]
        :param topk: top-k parameter for the Beamsearch algorithm. A lower top-k value will speed up the algorithm.  However, this will decrease the amount of candidate segmentations in a rank, defaults to 20
        :type topk: int, optional
        :param steps: steps parameter for the Beamsearch algorithm. A lower amount of steps will speed up the algorithm. However, the algorithm will never detect a number of words larger than amount of steps, defaults to 13
        :type steps: int, optional
        :param alpha: alpha parameter for the top-2 ensemble. It controls the weight given to the segmenter candidates. Reasonable values range from 0 to 1, defaults to 0.222
        :type alpha: float, optional
        :param beta: beta parameter for the top-2 ensemble. It controls the weight given to the reranker candidates. Reasonable values range from 0 to 1, defaults to 0.111
        :type beta: float, optional
        :param use_reranker: Whether or not to run the reranker, defaults to False
        :type use_reranker: bool, optional
        :param return_ranks: Return not just the segmented hashtags but also the a dictionary of the ranks, defaults to False
        :type return_ranks: bool, optional
        :return: A list of segmented words if return_ranks == False. A dictionary of the ranks and the segmented words if return_ranks == True.
        :rtype: Any
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

            replacement_dict.update({hashtag_key : hashtag_value})

        return replacement_dict

    def replace_hashtags(self, tweet, regex_pattern, replacement_dict):

        if not replacement_dict:
            return tweet

        tweet = regex_pattern.sub(lambda m: replacement_dict[m.group(0)], tweet)
        
        return tweet

    def segmented_tweet_generator(self, tweets, hashtags, hashtag_set, replacement_dict, flag=0):

        hashtag_set_index = { value:idx for idx, value in enumerate(hashtag_set)}
        replacement_pairs = [ (key, value) for key, value in replacement_dict.items() ]

        for idx, tweet_hashtags in enumerate(hashtags):
            
            tweet_dict = [ hashtag_set_index[hashtag] for hashtag in tweet_hashtags]
            tweet_dict = [ replacement_pairs[index] for index in tweet_dict ]
            tweet_dict = dict(tweet_dict)

            # Treats edge case: overlapping hashtags
            tweet_map = \
                map(re.escape, sorted(tweet_dict, key=len, reverse=True))
            
            regex_pattern = re.compile("|".join(tweet_map), flag)
            tweet = self.replace_hashtags(tweets[idx], regex_pattern, tweet_dict)
            yield tweet

    def build_hashtag_container(self, tweets: str, preprocessing_kwargs: dict = {}, segmenter_kwargs: dict = {} ):
        
        hashtags = self.extract_hashtags(tweets)

        hashtag_set = list(set(reduce(lambda x, y: x + y, hashtags)))
        
        word_segmenter_output = self.word_segmenter.predict(hashtag_set, **segmenter_kwargs)

        segmentations = word_segmenter_output.output

        replacement_dict = self.compile_dict(hashtag_set, segmentations, **preprocessing_kwargs)

        return HashtagContainer(hashtags, hashtag_set, replacement_dict), word_segmenter_output
  
    def segment(self, tweets: List[str], regex_flag: Any = 0, preprocessing_kwargs: dict = {}, segmenter_kwargs: dict = {} ):

        hashtag_container, word_segmenter_output = self.build_hashtag_container(tweets, preprocessing_kwargs, segmenter_kwargs)
        output = list(self.segmented_tweet_generator(tweets, *dataclasses.astuple(hashtag_container), flag=regex_flag))

        return TweetSegmenterOutput(
            word_segmenter_output = word_segmenter_output,
            output = output
        )
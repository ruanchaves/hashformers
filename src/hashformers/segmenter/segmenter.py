from hashformers.beamsearch.data_structures import enforce_prob_dict
from typing import List, Any
from ttp import ttp
from hashformers.segmenter.base_segmenter import BaseSegmenter
from hashformers.segmenter.regex_segmenter import RegexWordSegmenter
from hashformers.segmenter.data_structures import ( 
    WordSegmenterOutput, 
    TweetSegmenterOutput, 
    HashtagContainer
)
import re
from functools import reduce
import dataclasses
import pandas as pd

class WordSegmenterCascade(BaseSegmenter):

    def __init__(self, cascade_nodes):
        self.cascade_nodes = cascade_nodes

    def generate_pipeline(self, word_list):

        self.cascade_nodes[0].word_segmenter_kwargs.setdefault("return_ranks", True)

        first_ws_output = self.cascade_nodes[0].word_segmenter.segment(
            word_list, 
            **self.cascade_nodes[0].word_segmenter_kwargs)

        cascade_stack = [first_ws_output]
        pipeline = [first_ws_output]
        
        for idx in range(len(self.cascade_nodes)):

            self.cascade_nodes[idx].word_segmenter_kwargs.setdefault("return_ranks", True)

            if idx:
                previous_ws_output = cascade_stack.pop()
                for item in ["ensemble_rank", "reranker_rank", "segmenter_rank"]:
                    next_input = getattr(previous_ws_output, item)
                    if isinstance(next_input, pd.DataFrame):
                        break
                current_kwargs = self.cascade_nodes[idx].word_segmenter_kwargs
                if isinstance(next_input, pd.DataFrame):
                    current_kwargs.setdefault("segmenter_run", next_input)
                current_ws_output = self.cascade_nodes[idx].word_segmenter.segment(
                    word_list, 
                    **current_kwargs)
                cascade_stack.append(current_ws_output)
                pipeline.append(current_ws_output)

        return pipeline

    def segment(self, word_list, **kwargs):
        word_list = super().preprocess(word_list, **kwargs)
        return self.generate_pipeline(word_list)[-1]

class BaseWordSegmenter(BaseSegmenter):
    """A general-purpose word segmentation API.
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

    def segment(
            self,
            word_list: List[str],
            segmenter_run: Any = None,
            preprocessing_kwargs: dict = {},
            segmenter_kwargs: dict = {},
            ensembler_kwargs: dict = {},
            reranker_kwargs: dict = {},
            use_reranker: bool = True,
            use_ensembler: bool = True,
            return_ranks: bool = False) -> Any :
            
        word_list = super().preprocess(word_list, **preprocessing_kwargs)

        if not isinstance(segmenter_run, pd.DataFrame):
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
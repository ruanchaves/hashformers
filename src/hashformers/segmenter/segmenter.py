from hashformers.beamsearch.data_structures import enforce_prob_dict
from typing import Any
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
        """
        Initializes TwitterTextMatcher object with a Parser from ttp module.
        """
        self.parser = ttp.Parser()
    
    def __call__(self, tweets):
        """
        Makes the TwitterTextMatcher instance callable. It parses the given tweets and returns their tags.

        Args:
            tweets: A list of strings, where each string is a tweet.

        Returns:
            A list of hashtags for each tweet.
        """
        return [ self.parser.parse(x).tags for x in tweets ]
    
class TweetSegmenter(BaseSegmenter):

    def __init__(self, matcher=None, word_segmenter=None):
        """
        Initializes a TweetSegmenter instance with a TwitterTextMatcher and a WordSegmenter.

        Args:
            matcher (TwitterTextMatcher, optional): Instance of TwitterTextMatcher used for matching text in tweets. 
                Defaults to an instance of TwitterTextMatcher if not provided.
            word_segmenter (WordSegmenter, optional): Instance of WordSegmenter used for segmenting words in tweets. 
                Defaults to an instance of RegexWordSegmenter if not provided.
        """
        if matcher:
            self.matcher = matcher
        else:
            self.matcher = TwitterTextMatcher()

        if word_segmenter:
            self.word_segmenter = word_segmenter
        else:
            self.word_segmenter = RegexWordSegmenter()

    def extract_hashtags(self, tweets):
        """
        Extracts hashtags from the provided list of tweets.

        Args:
            tweets: A list of strings, where each string is a tweet.

        Returns:
            A list of hashtags extracted from each tweet.
        """
        return self.matcher(tweets)

    def compile_dict(self, hashtags, segmentations, hashtag_token=None, lower=False, separator=" ", hashtag_character="#"):
        """
        Compiles a dictionary mapping hashtags to their corresponding segmentations.

        Args:
            hashtags (list): List of hashtags extracted from tweets.
            segmentations (list): Corresponding segmentations of the hashtags.
            hashtag_token (str, optional): Token to prepend to the segmented hashtag value. If not provided, no token is prepended.
            lower (bool, optional): If True, converts the hashtag value to lowercase. Defaults to False.
            separator (str, optional): Separator used between hashtag_token and the value. Defaults to " ".
            hashtag_character (str, optional): Character representing a hashtag. Defaults to "#".

        Returns:
            dict: A dictionary mapping hashtags to their segmented versions.
        """
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
        """
        Replaces the hashtags in a tweet based on a provided replacement dictionary.

        Args:
            tweet (str): The tweet in which hashtags are to be replaced.
            regex_pattern (re.Pattern): Compiled regular expression pattern for matching hashtags in the tweet.
            replacement_dict (dict): Dictionary with original hashtags as keys and their replacements as values.

        Returns:
            str: The tweet with hashtags replaced.
        """
        if not replacement_dict:
            return tweet

        tweet = regex_pattern.sub(lambda m: replacement_dict[m.group(0)], tweet)
        
        return tweet

    def segmented_tweet_generator(self, tweets, hashtags, hashtag_set, replacement_dict, flag=0):
        """
        Yields segmented tweets from a provided list of tweets.

        Args:
            tweets (list): List of tweets to be segmented.
            hashtags (list): List of hashtags extracted from each tweet.
            hashtag_set (set): Set of unique hashtags extracted from all tweets.
            replacement_dict (dict): Dictionary with original hashtags as keys and their replacements as values.
            flag (int, optional): Flags for the regular expression compilation. Defaults to 0.

        Yields:
            str: Segmented version of each tweet.
        """
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
        """
        Constructs a HashtagContainer from a list of tweets.

        Args:
            tweets (list): List of tweets.
            preprocessing_kwargs (dict, optional): Keyword arguments for preprocessing. Defaults to an empty dictionary.
            segmenter_kwargs (dict, optional): Keyword arguments for the segmenter. Defaults to an empty dictionary.

        Returns:
            tuple: A tuple containing a HashtagContainer instance and the output from the word segmenter.
        """        
        hashtags = self.extract_hashtags(tweets)

        hashtag_set = list(set(reduce(lambda x, y: x + y, hashtags)))
        
        word_segmenter_output = self.word_segmenter.predict(hashtag_set, **segmenter_kwargs)

        segmentations = word_segmenter_output.output

        replacement_dict = self.compile_dict(hashtag_set, segmentations, **preprocessing_kwargs)

        return HashtagContainer(hashtags, hashtag_set, replacement_dict), word_segmenter_output
  
    def segment(self, tweets: list[str], regex_flag: Any = 0, preprocessing_kwargs: dict = {}, segmenter_kwargs: dict = {} ):
        """
        Segments a list of tweets into individual words and replaces the hashtags based on the preprocessing and segmenter configurations.

        Args:
            tweets (List[str]): List of tweets to be segmented.
            regex_flag (Any, optional): Regular expression flags used in replacing the hashtags. Defaults to 0.
            preprocessing_kwargs (dict, optional): Dictionary of keyword arguments used for preprocessing the tweets. Defaults to an empty dictionary.
            segmenter_kwargs (dict, optional): Dictionary of keyword arguments used for the WordSegmenter. Defaults to an empty dictionary.

        Returns:
            TweetSegmenterOutput: Contains the output of WordSegmenter and the segmented tweets.
        """
        hashtag_container, word_segmenter_output = self.build_hashtag_container(tweets, preprocessing_kwargs, segmenter_kwargs)
        output = list(self.segmented_tweet_generator(tweets, *dataclasses.astuple(hashtag_container), flag=regex_flag))

        return TweetSegmenterOutput(
            word_segmenter_output = word_segmenter_output,
            output = output
        )
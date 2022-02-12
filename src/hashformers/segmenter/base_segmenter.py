from hashformers.segmenter.data_structures import (
    WordSegmenterOutput,
    TweetSegmenterOutput
)

from collections.abc import Iterable

def coerce_segmenter_objects(method):
    def wrapper(self, inputs, *args, **kwargs):
        
        if isinstance(inputs, str):
            output = method(self, [inputs], *args, **kwargs)
        elif isinstance(inputs, Iterable):
            output = method(self, inputs, *args, **kwargs)
        else:
            raise NotImplementedError(str(type(inputs)))
        
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

class BaseSegmenter(object):

    @coerce_segmenter_objects
    def predict(self, inputs, *args, **kwargs):
        return self.segment(inputs, *args, **kwargs)
    
    def preprocess(self, inputs, lower=False, remove_hashtag=True, hashtag_character="#"):
        def preprocess_input(word):
            if lower:
                word = word.lower()
            if remove_hashtag:
                word = word.lstrip(hashtag_character)
            return word
        
        if isinstance(inputs, str):
            inputs = preprocess_input(inputs)
        elif isinstance(inputs, Iterable):
            inputs = [ preprocess_input(x) for x in inputs ]
        else:
            raise NotImplementedError(str(type(inputs)))
        
        return inputs
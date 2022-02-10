from hashformers.segmenter.dataclasses import (
    WordSegmenterOutput,
    TweetSegmenterOutput
)
from collections.abc import Iterable

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

class BaseSegmenter(object):

    @coerce_segmenter_objects
    def predict(self, *args, **kwargs):
        return self.segment(*args, **kwargs)
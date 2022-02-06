import hashformers
import pytest
import json
from hashformers import prune_segmenter_layers

import hashformers

with open("fixtures/test_boun_sample.txt", "r") as f1,\
    open("fixtures/word_Segmenters.json") as f2:

    test_boun_gold = f1.read().strip().split("\n")
    test_boun_hashtags = [ x.replace(" ", "") for x in test_boun_gold]
    word_segmenter_params = json.load(f2)


@pytest.fixture(scope="module", params=word_segmenter_params)
def word_segmenter(request):
    
    word_segmenter_class = request.param["class"]
    word_segmenter_init_kwargs = request.param["init_kwargs"]
    word_segmenter_predict_kwargs = request.param["predict_kwargs"]

    WordSegmenterClass = getattr(hashformers, word_segmenter_class)

    class PartialWordSegmenterClass(WordSegmenterClass):

        def __init__(self, **kwargs):
            return super().__init__(**kwargs)
        
        def predict(self, *args):
            super().predict(*args, **word_segmenter_predict_kwargs)

    ws = PartialWordSegmenterClass(**word_segmenter_init_kwargs)
    
    if request.param.get("prune", False):
        ws = prune_segmenter_layers(ws, layer_list=[0])

    return ws

def test_word_segmenter_output_format():
    
    model = word_segmenter()
    
    predictions = model.predict(test_boun_hashtags)

    predictions_chars = [ x.replace(" ", "") for x in predictions ]
    
    assert all([x == y for x,y in zip(test_boun_hashtags, predictions_chars)])
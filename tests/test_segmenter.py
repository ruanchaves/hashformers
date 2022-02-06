import hashformers
import pytest
import json
from hashformers import prune_segmenter_layers
from pathlib import Path
import hashformers
import os
test_data_dir = Path(__file__).parent.absolute()

with open(os.path.join(test_data_dir,"fixtures/test_boun_sample.txt"), "r") as f1,\
     open(os.path.join(test_data_dir,"fixtures/word_segmenters.json"), "r") as f2:

    test_boun_gold = f1.read().strip().split("\n")
    test_boun_hashtags = [ x.replace(" ", "") for x in test_boun_gold]
    word_segmenter_params = json.load(f2)

@pytest.fixture(scope="module", params=word_segmenter_params)
def word_segmenter(request):
    
    word_segmenter_class = request.param["class"]
    word_segmenter_init_kwargs = request.param["init_kwargs"]
    word_segmenter_predict_kwargs = request.param["predict_kwargs"]

    WordSegmenterClass = getattr(hashformers, word_segmenter_class)

    class WordSegmenterClassWrapper(WordSegmenterClass):

        def __init__(self, **kwargs):
            return super().__init__(**kwargs)
        
        def predict(self, *args):
            return super().predict(*args, **word_segmenter_predict_kwargs)

    WordSegmenterClassWrapper.__name__ = request.param["class"] + "ClassWrapper"

    ws = WordSegmenterClassWrapper(**word_segmenter_init_kwargs)
    
    if request.param.get("prune", False):
        ws = prune_segmenter_layers(ws, layer_list=[0])

    return ws

def test_word_segmenter_output_format(word_segmenter):
    
    predictions = word_segmenter.predict(test_boun_hashtags).output

    predictions_chars = [ x.replace(" ", "") for x in predictions ]
    
    assert all([x == y for x,y in zip(test_boun_hashtags, predictions_chars)])
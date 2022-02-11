from hashformers.segmenter.base_segmenter import BaseSegmenter
from typing import List
import re

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

    def segment(self, inputs: List[str], **kwargs):
        inputs = super().preprocess(inputs, **kwargs)
        return list(self.segmentation_generator(inputs))
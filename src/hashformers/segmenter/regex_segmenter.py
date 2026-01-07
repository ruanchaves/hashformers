from hashformers.segmenter.base_segmenter import BaseSegmenter
import re

class RegexWordSegmenter(BaseSegmenter):
    """
    A subclass of BaseSegmenter which uses regex rules to perform segmentation.

    Args:
        regex_rules (list of str, optional): List of regex rules used for segmentation. 
        If none are provided, it uses the default rule: [r'([A-Z]+)'].

    Attributes:
        regex_rules (list of _sre.SRE_Pattern): List of compiled regex rules used for segmentation.
    """
    def __init__(self,regex_rules=None):
        """
        Initializes the RegexWordSegmenter with given or default regex rules and compiles them.
        """
        if not regex_rules:
            regex_rules = [r'([A-Z]+)']
        self.regex_rules = [
            re.compile(x) for x in regex_rules
        ]

    def segment_word(self, rule, word):
        """
        Segments a word based on a given regex rule.

        Args:
            rule (_sre.SRE_Pattern): The compiled regex rule to be used for segmentation.
            word (str): The word to be segmented.

        Returns:
            str: The segmented word.
        """
        return rule.sub(r' \1', word).strip()

    def segmentation_generator(self, word_list):
        """
        A generator that iterates over the word list and yields segmented words 
        based on the regex rules.

        Args:
            word_list (list of str): The list of words to be segmented.

        Yields:
            str: The segmented word.
        """
        for rule in self.regex_rules:
            for idx, word in enumerate(word_list):
                yield self.segment_word(rule, word)

    def segment(self, inputs: list[str], **kwargs):
        """
        Segments a list of strings based on the regex rules. Before segmentation, 
        the inputs are preprocessed using the inherited preprocess method.

        Args:
            inputs (List[str]): The list of strings to be segmented.
            **kwargs: Arbitrary keyword arguments for the inherited preprocess method.

        Returns:
            list of str: The segmented inputs.
        """
        inputs = super().preprocess(inputs, **kwargs)
        return list(self.segmentation_generator(inputs))
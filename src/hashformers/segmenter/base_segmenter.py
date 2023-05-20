from hashformers.segmenter.data_structures import (
    WordSegmenterOutput,
    TweetSegmenterOutput
)

from collections.abc import Iterable

def coerce_segmenter_objects(method):
    """
    A decorator function that ensures the returned value from the decorated method is of certain types 
    or converts the returned value to one of the allowed types.

    It also handles different types of 'inputs' passed to the decorated method. It checks whether the input is a string 
    or an iterable, and in the case of unsupported input type, raises a NotImplementedError.

    Args:
        method (function): The method to be decorated.

    Returns:
        function: The decorated function which enforces specific output types and handles different input types.
    """
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
    """
    Base class for text segmenter objects.
    """
    @coerce_segmenter_objects
    def predict(self, inputs, *args, **kwargs):
        """
        Predict method that delegates to the segment method.
        It is decorated with coerce_segmenter_objects to handle different input types and enforce output type.

        Args:
            inputs (str or Iterable): The inputs to be segmented.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            WordSegmenterOutput or TweetSegmenterOutput: The output from the segment method.
        """
        return self.segment(inputs, *args, **kwargs)
    
    def segment(self, inputs, *args, **kwargs):
        """
        Abstract method for segmentation. Should be implemented in a child class.

        Args:
            inputs (str or Iterable): The inputs to be segmented.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            NotImplementedError: If this method is not overridden in a child class.
        """
        raise NotImplementedError("This method should be implemented in a child class.")

    def preprocess(self, inputs, lower=False, remove_hashtag=True, hashtag_character="#"):
        """
        Preprocesses the inputs based on the given parameters.

        Args:
            inputs (str or Iterable): The inputs to be preprocessed.
            lower (bool, optional): Whether to convert the inputs to lower case. Defaults to False.
            remove_hashtag (bool, optional): Whether to remove the hashtag character from the inputs. Defaults to True.
            hashtag_character (str, optional): The hashtag character to be removed. Defaults to "#".

        Returns:
            str or list of str: The preprocessed inputs.

        Raises:
            NotImplementedError: If the type of inputs is neither str nor Iterable.
        """
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
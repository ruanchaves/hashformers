from collections.abc import Iterable

from hashformers.segmenter.data_structures import WordSegmenterOutput


def coerce_segmenter_objects(method):
    """Coerce segmenter inputs and outputs to the public container type.

    Args:
        method: Segmenter method to wrap.

    Returns:
        A wrapper accepting one string or an iterable of strings and returning
        ``WordSegmenterOutput``.
    """

    def wrapper(self, inputs, *args, **kwargs):
        if isinstance(inputs, str):
            output = method(self, [inputs], *args, **kwargs)
        elif isinstance(inputs, Iterable):
            output = method(self, inputs, *args, **kwargs)
        else:
            raise NotImplementedError(str(type(inputs)))

        if isinstance(output, WordSegmenterOutput):
            return output
        if isinstance(output, str):
            return WordSegmenterOutput(output=[output])
        if isinstance(output, Iterable):
            return WordSegmenterOutput(output=output)

    return wrapper


class BaseSegmenter(object):
    """Base class for text segmenter objects."""

    @coerce_segmenter_objects
    def predict(self, inputs, *args, **kwargs):
        """Delegate prediction to ``segment`` and normalize its output.

        Args:
            inputs: One string or an iterable of strings to segment.
            *args: Positional arguments forwarded to ``segment``.
            **kwargs: Keyword arguments forwarded to ``segment``.

        Returns:
            WordSegmenterOutput: Normalized segmenter output.
        """
        return self.segment(inputs, *args, **kwargs)

    def segment(self, inputs, *args, **kwargs):
        """Segment inputs in a subclass implementation.

        Raises:
            NotImplementedError: Always; subclasses must override this method.
        """
        raise NotImplementedError("This method should be implemented in a child class.")

    def preprocess(
        self,
        inputs,
        lower=False,
        remove_hashtag=True,
        hashtag_character="#",
    ):
        """Apply shared casing and leading-hashtag preprocessing.

        Args:
            inputs: One string or an iterable of strings.
            lower: Whether to lowercase every input.
            remove_hashtag: Whether to strip leading hashtag characters.
            hashtag_character: Character stripped from the left edge.

        Returns:
            A processed string or list of strings matching the input shape.

        Raises:
            NotImplementedError: If ``inputs`` is neither a string nor iterable.
        """

        def preprocess_input(word):
            if lower:
                word = word.lower()
            if remove_hashtag:
                word = word.lstrip(hashtag_character)
            return word

        if isinstance(inputs, str):
            return preprocess_input(inputs)
        if isinstance(inputs, Iterable):
            return [preprocess_input(value) for value in inputs]
        raise NotImplementedError(str(type(inputs)))

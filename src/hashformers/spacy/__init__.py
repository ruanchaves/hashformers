"""
spaCy integration for hashformers.

This module provides a spaCy pipeline component for word segmentation
using hashformers' TransformerWordSegmenter.

Example usage:
    >>> import spacy
    >>> from hashformers.spacy import HashformersComponent
    >>> 
    >>> nlp = spacy.blank("en")
    >>> nlp.add_pipe("hashformers", config={"model": "distilgpt2"})
    >>> 
    >>> doc = nlp("#weneedanationalpark")
    >>> print(doc._.segmented)  # "we need a national park"
"""

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from hashformers import TransformerWordSegmenter


class HashformersComponent:
    """
    A spaCy pipeline component that performs word segmentation on text.
    
    This component uses hashformers' TransformerWordSegmenter to segment
    text (typically hashtags) into individual words. The segmented text
    is stored in the doc._.segmented extension.
    
    Attributes:
        nlp: The spaCy Language object.
        name: The component name.
        segmenter: The TransformerWordSegmenter instance.
    """
    
    def __init__(
        self,
        nlp,
        name: str,
        model: str = "distilgpt2",
        device: str = "cuda",
        gpu_batch_size: int = 64
    ):
        """
        Initialize the HashformersComponent.
        
        Args:
            nlp: The spaCy Language object.
            name: The component name in the pipeline.
            model: The Hugging Face model name or path for segmentation.
                Defaults to "distilgpt2".
            device: Device to run the model on ("cuda" or "cpu").
                Defaults to "cuda".
            gpu_batch_size: Batch size for GPU processing.
                Defaults to 64.
        """
        self.nlp = nlp
        self.name = name
        self.segmenter = TransformerWordSegmenter(
            segmenter_model_name_or_path=model,
            segmenter_device=device,
            segmenter_gpu_batch_size=gpu_batch_size
        )
        
        # Register the custom extension if not already registered
        if not Doc.has_extension("segmented"):
            Doc.set_extension("segmented", default=None)
    
    def __call__(self, doc: "Doc") -> "Doc":
        """
        Process a spaCy Doc and add segmented text to doc._.segmented.
        
        The component strips any leading '#' character from the text
        before segmentation.
        
        Args:
            doc: The spaCy Doc object to process.
            
        Returns:
            The processed Doc with segmented text in doc._.segmented.
        """
        # Get the text, stripping any leading '#' characters
        text = doc.text.lstrip('#')
        
        if text:
            # Perform segmentation
            result = self.segmenter.segment([text])
            doc._.segmented = result[0] if result else text
        else:
            doc._.segmented = ""
        
        return doc


# Register the component with spaCy if spaCy is available
if SPACY_AVAILABLE:
    @Language.factory(
        "hashformers",
        default_config={
            "model": "distilgpt2",
            "device": "cuda",
            "gpu_batch_size": 64
        }
    )
    def create_hashformers_component(
        nlp: Language,
        name: str,
        model: str,
        device: str,
        gpu_batch_size: int
    ) -> HashformersComponent:
        """
        Factory function to create a HashformersComponent for spaCy pipelines.
        
        Args:
            nlp: The spaCy Language object.
            name: The component name.
            model: The Hugging Face model name or path.
            device: Device to run the model on.
            gpu_batch_size: Batch size for GPU processing.
            
        Returns:
            A configured HashformersComponent instance.
        """
        return HashformersComponent(
            nlp=nlp,
            name=name,
            model=model,
            device=device,
            gpu_batch_size=gpu_batch_size
        )


__all__ = ["HashformersComponent"]

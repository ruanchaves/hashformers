"""spaCy pipeline component for hashformers.

This module provides a spaCy pipeline component that uses hashformers
for word segmentation during NLP processing.

HASH-402: Create spaCy Pipeline Component
"""

from typing import Any, Callable, Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)

# Try to import spaCy - it's an optional dependency
try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Language = None
    Doc = None


def create_hashformers_component(
    nlp: Any,
    name: str,
    segmenter_model: str = "gpt2",
    segmenter_type: str = "incremental",
    reranker_model: Optional[str] = None,
    reranker_type: Optional[str] = None,
    segment_hashtags: bool = True,
    segment_identifiers: bool = False,
    device: str = "cuda"
) -> "HashformersComponent":
    """Factory function for creating the hashformers spaCy component.
    
    This function is registered with spaCy's Language.factory decorator
    and is called when users add the component to their pipeline.
    
    Args:
        nlp: The spaCy Language object.
        name: The component name.
        segmenter_model: Name or path of the segmenter model.
        segmenter_type: Type of segmenter model.
        reranker_model: Optional reranker model.
        reranker_type: Type of reranker model.
        segment_hashtags: Whether to segment hashtags.
        segment_identifiers: Whether to segment camelCase/PascalCase identifiers.
        device: Device to run models on.
        
    Returns:
        A HashformersComponent instance.
    """
    return HashformersComponent(
        nlp=nlp,
        name=name,
        segmenter_model=segmenter_model,
        segmenter_type=segmenter_type,
        reranker_model=reranker_model,
        reranker_type=reranker_type,
        segment_hashtags=segment_hashtags,
        segment_identifiers=segment_identifiers,
        device=device
    )


# Register the component factory with spaCy if available
if SPACY_AVAILABLE and Language is not None:
    @Language.factory(
        "hashformers",
        default_config={
            "segmenter_model": "gpt2",
            "segmenter_type": "incremental",
            "reranker_model": None,
            "reranker_type": None,
            "segment_hashtags": True,
            "segment_identifiers": False,
            "device": "cuda"
        }
    )
    def _create_component(
        nlp: Language,
        name: str,
        segmenter_model: str,
        segmenter_type: str,
        reranker_model: Optional[str],
        reranker_type: Optional[str],
        segment_hashtags: bool,
        segment_identifiers: bool,
        device: str
    ):
        return create_hashformers_component(
            nlp=nlp,
            name=name,
            segmenter_model=segmenter_model,
            segmenter_type=segmenter_type,
            reranker_model=reranker_model,
            reranker_type=reranker_type,
            segment_hashtags=segment_hashtags,
            segment_identifiers=segment_identifiers,
            device=device
        )


class HashformersComponent:
    """spaCy pipeline component for word segmentation with hashformers.
    
    This component can be added to a spaCy pipeline to segment concatenated
    text like hashtags and camelCase identifiers.
    
    Example:
        >>> import spacy
        >>> import hashformers.integrations.spacy_component  # registers component
        >>> 
        >>> nlp = spacy.blank("en")
        >>> nlp.add_pipe("hashformers", config={"segmenter_model": "gpt2"})
        >>> 
        >>> doc = nlp("Check out #MakeAmericaGreatAgain")
        >>> print(doc._.segmented_text)
        'Check out Make America Great Again'
    
    Attributes:
        name: The component name in the pipeline.
        segmenter: The hashformers word segmenter (lazy-loaded).
    """
    
    def __init__(
        self,
        nlp: Any,
        name: str,
        segmenter_model: str = "gpt2",
        segmenter_type: str = "incremental",
        reranker_model: Optional[str] = None,
        reranker_type: Optional[str] = None,
        segment_hashtags: bool = True,
        segment_identifiers: bool = False,
        device: str = "cuda"
    ):
        self.nlp = nlp
        self.name = name
        self.segmenter_model = segmenter_model
        self.segmenter_type = segmenter_type
        self.reranker_model = reranker_model
        self.reranker_type = reranker_type
        self.segment_hashtags = segment_hashtags
        self.segment_identifiers = segment_identifiers
        self.device = device
        self._segmenter = None
        
        # Regex patterns
        self._hashtag_pattern = re.compile(r'#(\w+)')
        self._camel_case_pattern = re.compile(r'([a-z])([A-Z])')
        self._identifier_pattern = re.compile(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:[A-Z][a-z]+)+)\b')
        
        # Register custom Doc extension if not already registered
        if SPACY_AVAILABLE and Doc is not None:
            if not Doc.has_extension("segmented_text"):
                Doc.set_extension("segmented_text", default=None)
            if not Doc.has_extension("hashformers_segments"):
                Doc.set_extension("hashformers_segments", default={})
    
    @property
    def segmenter(self):
        """Lazy-load the segmenter."""
        if self._segmenter is None:
            from hashformers import TransformerWordSegmenter
            self._segmenter = TransformerWordSegmenter(
                segmenter_model_name_or_path=self.segmenter_model,
                segmenter_model_type=self.segmenter_type,
                reranker_model_name_or_path=self.reranker_model,
                reranker_model_type=self.reranker_type
            )
        return self._segmenter
    
    def __call__(self, doc: Any) -> Any:
        """Process a spaCy Doc object.
        
        Args:
            doc: A spaCy Doc object.
            
        Returns:
            The processed Doc with segmented text in custom extensions.
        """
        if not SPACY_AVAILABLE:
            logger.warning("spaCy is not installed. Returning doc unchanged.")
            return doc
        
        text = doc.text
        segments = {}
        
        # Segment hashtags
        if self.segment_hashtags:
            hashtags = self._hashtag_pattern.findall(text)
            if hashtags:
                segmented = self.segmenter.segment(hashtags)
                for original, seg in zip(hashtags, segmented):
                    segments[f"#{original}"] = seg
                    text = text.replace(f"#{original}", seg)
        
        # Segment identifiers
        if self.segment_identifiers:
            identifiers = self._identifier_pattern.findall(text)
            if identifiers:
                segmented = self.segmenter.segment(identifiers)
                for original, seg in zip(identifiers, segmented):
                    segments[original] = seg
                    text = text.replace(original, seg)
        
        # Store results in Doc extensions
        doc._.segmented_text = text
        doc._.hashformers_segments = segments
        
        return doc

"""Haystack 2.0 integration for hashformers.

This module provides a Haystack 2.0 component that uses hashformers
for word segmentation in document processing pipelines.

HASH-408: Create Haystack 2.0 Component for Privacy-Focused European market
"""

from typing import Any, Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)

# Try to import Haystack - it's an optional dependency
try:
    from haystack import Document, component
    from haystack.components.preprocessors import DocumentCleaner
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    # Create stub decorator and classes for when Haystack isn't installed
    def component(cls):
        return cls
    class Document:
        def __init__(self, content: str = "", meta: dict = None, **kwargs):
            self.content = content
            self.meta = meta or {}


@component
class HashformersSegmenter:
    """Haystack 2.0 component for word segmentation with hashformers.
    
    This component can be used in Haystack pipelines to segment concatenated
    text like hashtags, URLs, and code identifiers into readable words.
    Compatible with Haystack.Pipeline for privacy-focused European deployments.
    
    Example:
        >>> from haystack import Pipeline
        >>> from haystack.components.converters import TextFileToDocument
        >>> from hashformers.integrations.haystack import HashformersSegmenter
        >>> 
        >>> segmenter = HashformersSegmenter(
        ...     segmenter_model="gpt2",
        ...     segment_hashtags=True
        ... )
        >>> 
        >>> pipeline = Pipeline()
        >>> pipeline.add_component("converter", TextFileToDocument())
        >>> pipeline.add_component("segmenter", segmenter)
        >>> pipeline.connect("converter", "segmenter")
        >>> 
        >>> result = pipeline.run({"converter": {"sources": ["tweets.txt"]}})
    
    Args:
        segmenter_model: Name or path of the segmenter model. Default is "gpt2".
        segmenter_type: Type of segmenter model ("incremental", "masked", "seq2seq").
        reranker_model: Optional reranker model name or path.
        reranker_type: Type of reranker model.
        device: Device to run models on ("cuda" or "cpu").
        segment_hashtags: If True, segment hashtags. Default is True.
        segment_urls: If True, segment URL slugs. Default is False.
        segment_camel_case: If True, segment camelCase identifiers. Default is False.
    """
    
    def __init__(
        self,
        segmenter_model: str = "gpt2",
        segmenter_type: str = "incremental",
        reranker_model: Optional[str] = None,
        reranker_type: Optional[str] = None,
        device: str = "cuda",
        segment_hashtags: bool = True,
        segment_urls: bool = False,
        segment_camel_case: bool = False
    ):
        if not HAYSTACK_AVAILABLE:
            logger.warning(
                "Haystack is not installed. Install with: pip install haystack-ai"
            )
        
        self.segmenter_model = segmenter_model
        self.segmenter_type = segmenter_type
        self.reranker_model = reranker_model
        self.reranker_type = reranker_type
        self.device = device
        self.segment_hashtags = segment_hashtags
        self.segment_urls = segment_urls
        self.segment_camel_case = segment_camel_case
        self._segmenter = None
        
        # Regex patterns
        self._hashtag_pattern = re.compile(r'#(\w+)')
        self._url_slug_pattern = re.compile(
            r'(?<=/|-)([a-z0-9]+(?:-[a-z0-9]+)+)(?=/|$)',
            re.IGNORECASE
        )
        self._camel_case_pattern = re.compile(
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:[A-Z][a-z]+)+)\b'
        )
    
    @property
    def segmenter(self):
        """Lazy-load the segmenter to avoid loading models until needed."""
        if self._segmenter is None:
            from hashformers import TransformerWordSegmenter
            self._segmenter = TransformerWordSegmenter(
                segmenter_model_name_or_path=self.segmenter_model,
                segmenter_model_type=self.segmenter_type,
                reranker_model_name_or_path=self.reranker_model,
                reranker_model_type=self.reranker_type
            )
        return self._segmenter
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Process documents by segmenting concatenated text.
        
        Args:
            documents: List of Haystack Document objects.
            
        Returns:
            Dictionary with "documents" key containing processed documents.
        """
        if not HAYSTACK_AVAILABLE:
            logger.warning("Haystack is not installed. Returning documents unchanged.")
            return {"documents": documents}
        
        processed_docs = []
        
        for doc in documents:
            if doc.content:
                new_content = self._process_text(doc.content)
                new_meta = dict(doc.meta) if doc.meta else {}
                new_meta["hashformers_processed"] = True
                
                processed_docs.append(Document(
                    content=new_content,
                    meta=new_meta
                ))
            else:
                processed_docs.append(doc)
        
        return {"documents": processed_docs}
    
    def _process_text(self, text: str) -> str:
        """Process text by segmenting target patterns.
        
        Args:
            text: Input text to process.
            
        Returns:
            Text with segmented patterns.
        """
        result = text
        
        if self.segment_hashtags:
            result = self._segment_hashtags(result)
        
        if self.segment_urls:
            result = self._segment_url_slugs(result)
        
        if self.segment_camel_case:
            result = self._segment_camel_case(result)
        
        return result
    
    def _segment_hashtags(self, text: str) -> str:
        """Find and segment hashtags in text."""
        hashtags = self._hashtag_pattern.findall(text)
        
        if not hashtags:
            return text
        
        segmented = self.segmenter.segment(hashtags)
        
        result = text
        for original, segmented_text in zip(hashtags, segmented):
            result = result.replace(f"#{original}", segmented_text)
        
        return result
    
    def _segment_url_slugs(self, text: str) -> str:
        """Find and segment URL slugs in text."""
        slugs = self._url_slug_pattern.findall(text)
        
        if not slugs:
            return text
        
        segmented = self.segmenter.segment(slugs)
        
        result = text
        for original, segmented_text in zip(slugs, segmented):
            result = result.replace(original, segmented_text)
        
        return result
    
    def _segment_camel_case(self, text: str) -> str:
        """Find and segment camelCase identifiers."""
        identifiers = self._camel_case_pattern.findall(text)
        
        if not identifiers:
            return text
        
        lowercased = [id.lower() for id in identifiers]
        segmented = self.segmenter.segment(lowercased)
        
        result = text
        for original, segmented_text in zip(identifiers, segmented):
            result = result.replace(original, segmented_text)
        
        return result


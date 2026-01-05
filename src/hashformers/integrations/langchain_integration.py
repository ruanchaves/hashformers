"""LangChain integration for hashformers.

This module provides a LangChain Document Transformer that uses hashformers
for word segmentation in document processing pipelines.

HASH-401: Create LangChain Document Transformer
"""

from typing import Any, List, Optional, Sequence
import logging
import re

logger = logging.getLogger(__name__)

# Try to import LangChain - it's an optional dependency
try:
    from langchain_core.documents import BaseDocumentTransformer, Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create stub classes for when LangChain isn't installed
    class BaseDocumentTransformer:
        pass
    class Document:
        def __init__(self, page_content: str = "", metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


class HashformersTransformer(BaseDocumentTransformer):
    """LangChain Document Transformer using hashformers for word segmentation.
    
    This transformer processes documents to segment concatenated text like
    hashtags, URLs, and code identifiers into readable words.
    
    Example:
        >>> from hashformers.integrations import HashformersTransformer
        >>> from langchain.schema import Document
        >>> 
        >>> transformer = HashformersTransformer(
        ...     segmenter_model="gpt2",
        ...     segmenter_type="incremental"
        ... )
        >>> docs = [Document(page_content="#weneedanationalpark")]
        >>> result = transformer.transform_documents(docs)
        >>> print(result[0].page_content)
        'we need a national park'
    
    Args:
        segmenter_model: Name or path of the segmenter model. Default is "gpt2".
        segmenter_type: Type of segmenter model ("incremental", "masked", "seq2seq").
        reranker_model: Optional reranker model name or path.
        reranker_type: Type of reranker model.
        device: Device to run models on ("cuda" or "cpu").
        extract_hashtags: If True, only segment hashtags in text. Default is True.
        extract_urls: If True, segment URL-like text. Default is False.
    """
    
    def __init__(
        self,
        segmenter_model: str = "gpt2",
        segmenter_type: str = "incremental",
        reranker_model: Optional[str] = None,
        reranker_type: Optional[str] = None,
        device: str = "cuda",
        extract_hashtags: bool = True,
        extract_urls: bool = False,
        **kwargs: Any
    ):
        if not LANGCHAIN_AVAILABLE:
            logger.warning(
                "LangChain is not installed. Install with: pip install langchain-core"
            )
        
        self.segmenter_model = segmenter_model
        self.segmenter_type = segmenter_type
        self.reranker_model = reranker_model
        self.reranker_type = reranker_type
        self.device = device
        self.extract_hashtags = extract_hashtags
        self.extract_urls = extract_urls
        self._segmenter = None
        
        # Regex patterns for extraction
        self._hashtag_pattern = re.compile(r'#(\w+)')
        self._url_slug_pattern = re.compile(r'(?<=/|-)([a-z0-9]+(?:-[a-z0-9]+)+)(?=/|$)', re.IGNORECASE)
    
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
    
    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any
    ) -> Sequence[Document]:
        """Transform documents by segmenting concatenated text.
        
        Args:
            documents: Sequence of LangChain Document objects.
            **kwargs: Additional keyword arguments (unused).
            
        Returns:
            Sequence of transformed Document objects with segmented text.
        """
        transformed = []
        
        for doc in documents:
            new_content = self._process_text(doc.page_content)
            new_metadata = doc.metadata.copy() if doc.metadata else {}
            new_metadata["hashformers_processed"] = True
            
            transformed.append(Document(
                page_content=new_content,
                metadata=new_metadata
            ))
        
        return transformed
    
    async def atransform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any
    ) -> Sequence[Document]:
        """Async version of transform_documents.
        
        Currently runs synchronously as hashformers doesn't have async support.
        """
        return self.transform_documents(documents, **kwargs)
    
    def _process_text(self, text: str) -> str:
        """Process text by extracting and segmenting target patterns.
        
        Args:
            text: Input text to process.
            
        Returns:
            Text with segmented hashtags/URLs.
        """
        result = text
        
        if self.extract_hashtags:
            result = self._segment_hashtags(result)
        
        if self.extract_urls:
            result = self._segment_url_slugs(result)
        
        return result
    
    def _segment_hashtags(self, text: str) -> str:
        """Find and segment hashtags in text."""
        hashtags = self._hashtag_pattern.findall(text)
        
        if not hashtags:
            return text
        
        # Segment all hashtags at once for efficiency
        segmented = self.segmenter.segment(hashtags)
        
        # Replace in text
        result = text
        for original, segmented_text in zip(hashtags, segmented):
            result = result.replace(f"#{original}", segmented_text)
        
        return result
    
    def _segment_url_slugs(self, text: str) -> str:
        """Find and segment URL slugs in text."""
        slugs = self._url_slug_pattern.findall(text)
        
        if not slugs:
            return text
        
        # Segment all slugs at once
        segmented = self.segmenter.segment(slugs)
        
        # Replace in text
        result = text
        for original, segmented_text in zip(slugs, segmented):
            result = result.replace(original, segmented_text)
        
        return result

"""LlamaIndex integration for hashformers.

This module provides a LlamaIndex TransformComponent that uses hashformers
for cleaning and segmenting text before embedding in RAG pipelines.

HASH-407: Create LlamaIndex Node Parser for RAG data cleaning
"""

from typing import Any, Callable, List, Optional, Sequence
import logging
import re

logger = logging.getLogger(__name__)

# Try to import LlamaIndex - it's an optional dependency
try:
    from llama_index.core.schema import BaseNode, TextNode, TransformComponent
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Create stub classes for when LlamaIndex isn't installed
    class TransformComponent:
        pass
    class BaseNode:
        pass
    class TextNode:
        def __init__(self, text: str = "", metadata: dict = None, **kwargs):
            self.text = text
            self.metadata = metadata or {}


class HashformersCleaner(TransformComponent):
    """LlamaIndex TransformComponent for cleaning text with hashformers.
    
    This component processes nodes to segment concatenated text like
    hashtags, URLs, and code identifiers into readable words before
    embedding, improving RAG retrieval quality.
    
    Example:
        >>> from llama_index.core.ingestion import IngestionPipeline
        >>> from llama_index.core.node_parser import SentenceSplitter
        >>> from hashformers.integrations.llamaindex import HashformersCleaner
        >>> 
        >>> cleaner = HashformersCleaner(
        ...     segmenter_model="gpt2",
        ...     clean_hashtags=True,
        ...     clean_camel_case=True
        ... )
        >>> 
        >>> pipeline = IngestionPipeline(
        ...     transformations=[
        ...         SentenceSplitter(),
        ...         cleaner,
        ...     ]
        ... )
        >>> 
        >>> nodes = pipeline.run(documents=documents)
    
    Args:
        segmenter_model: Name or path of the segmenter model. Default is "gpt2".
        segmenter_type: Type of segmenter model ("incremental", "masked", "seq2seq").
        reranker_model: Optional reranker model name or path.
        reranker_type: Type of reranker model.
        device: Device to run models on ("cuda" or "cpu").
        clean_hashtags: If True, segment hashtags. Default is True.
        clean_camel_case: If True, segment camelCase identifiers. Default is False.
        clean_underscored: If True, segment snake_case identifiers. Default is False.
    """
    
    def __init__(
        self,
        segmenter_model: str = "gpt2",
        segmenter_type: str = "incremental",
        reranker_model: Optional[str] = None,
        reranker_type: Optional[str] = None,
        device: str = "cuda",
        clean_hashtags: bool = True,
        clean_camel_case: bool = False,
        clean_underscored: bool = False
    ):
        if not LLAMAINDEX_AVAILABLE:
            logger.warning(
                "LlamaIndex is not installed. Install with: pip install llama-index"
            )
        
        self.segmenter_model = segmenter_model
        self.segmenter_type = segmenter_type
        self.reranker_model = reranker_model
        self.reranker_type = reranker_type
        self.device = device
        self.clean_hashtags = clean_hashtags
        self.clean_camel_case = clean_camel_case
        self.clean_underscored = clean_underscored
        self._segmenter = None
        
        # Regex patterns
        self._hashtag_pattern = re.compile(r'#(\w+)')
        self._camel_case_pattern = re.compile(
            r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+|[a-z]+(?:[A-Z][a-z]+)+)\b'
        )
        self._underscored_pattern = re.compile(r'\b([a-z]+(?:_[a-z]+)+)\b')
    
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
    
    def __call__(
        self,
        nodes: Sequence[BaseNode],
        **kwargs: Any
    ) -> List[BaseNode]:
        """Transform nodes by cleaning and segmenting text.
        
        Args:
            nodes: Sequence of LlamaIndex nodes to process.
            **kwargs: Additional keyword arguments (unused).
            
        Returns:
            List of processed nodes with cleaned text.
        """
        if not LLAMAINDEX_AVAILABLE:
            logger.warning("LlamaIndex is not installed. Returning nodes unchanged.")
            return list(nodes)
        
        processed_nodes = []
        
        for node in nodes:
            if hasattr(node, 'text') and node.text:
                cleaned_text = self._clean_text(node.text)
                
                # Create a new node with cleaned text
                new_metadata = dict(node.metadata) if node.metadata else {}
                new_metadata["hashformers_cleaned"] = True
                
                # Handle different node types
                if isinstance(node, TextNode):
                    new_node = TextNode(
                        text=cleaned_text,
                        metadata=new_metadata,
                        id_=node.id_ if hasattr(node, 'id_') else None,
                        embedding=node.embedding if hasattr(node, 'embedding') else None,
                        relationships=node.relationships if hasattr(node, 'relationships') else {}
                    )
                else:
                    # For other node types, try to preserve attributes
                    new_node = node.copy()
                    new_node.text = cleaned_text
                    new_node.metadata = new_metadata
                
                processed_nodes.append(new_node)
            else:
                processed_nodes.append(node)
        
        return processed_nodes
    
    def _clean_text(self, text: str) -> str:
        """Clean text by segmenting concatenated patterns.
        
        Args:
            text: Input text to clean.
            
        Returns:
            Cleaned text with segmented patterns.
        """
        result = text
        
        if self.clean_hashtags:
            result = self._segment_hashtags(result)
        
        if self.clean_camel_case:
            result = self._segment_camel_case(result)
        
        if self.clean_underscored:
            result = self._segment_underscored(result)
        
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
    
    def _segment_camel_case(self, text: str) -> str:
        """Find and segment camelCase identifiers."""
        identifiers = self._camel_case_pattern.findall(text)
        
        if not identifiers:
            return text
        
        # Remove case for segmentation
        lowercased = [id.lower() for id in identifiers]
        segmented = self.segmenter.segment(lowercased)
        
        # Replace in text
        result = text
        for original, segmented_text in zip(identifiers, segmented):
            result = result.replace(original, segmented_text)
        
        return result
    
    def _segment_underscored(self, text: str) -> str:
        """Find and segment snake_case identifiers."""
        identifiers = self._underscored_pattern.findall(text)
        
        if not identifiers:
            return text
        
        # Replace underscores with spaces
        result = text
        for identifier in identifiers:
            result = result.replace(identifier, identifier.replace('_', ' '))
        
        return result


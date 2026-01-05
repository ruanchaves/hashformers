"""Hashformers integrations with external libraries.

This module provides integrations with popular NLP frameworks:
- LangChain: Document transformers for RAG pipelines
- spaCy: Pipeline components for NLP workflows
- PySpark: ML Pipeline transformers for Databricks/AWS EMR
- LlamaIndex: TransformComponents for RAG data cleaning
- Haystack: Pipeline components for privacy-focused deployments
"""

from hashformers.integrations.langchain_integration import HashformersTransformer
from hashformers.integrations.spacy_component import create_hashformers_component
from hashformers.integrations.spark import SparkHashformer
from hashformers.integrations.llamaindex import HashformersCleaner
from hashformers.integrations.haystack import HashformersSegmenter

__all__ = [
    "HashformersTransformer",
    "create_hashformers_component",
    "SparkHashformer",
    "HashformersCleaner",
    "HashformersSegmenter",
]

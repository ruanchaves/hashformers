"""Tests for integration modules.

HASH-409: Improve Test Coverage - Phase 2.1
Tests for hashformers.integrations
"""

import pytest


class TestLangchainIntegration:
    """Test the LangChain integration module."""

    def test_import_module(self):
        """langchain_integration module should be importable."""
        from hashformers.integrations import langchain_integration
        assert langchain_integration is not None

    def test_transformer_class_exists(self):
        """HashformersTransformer class should exist."""
        from hashformers.integrations.langchain_integration import HashformersTransformer
        assert HashformersTransformer is not None

    def test_transformer_init_defaults(self):
        """HashformersTransformer should initialize with defaults."""
        from hashformers.integrations.langchain_integration import HashformersTransformer
        transformer = HashformersTransformer()
        assert transformer.segmenter_model == "gpt2"
        assert transformer.segmenter_type == "incremental"
        assert transformer.device == "cuda"

    def test_transformer_init_custom(self):
        """HashformersTransformer should accept custom parameters."""
        from hashformers.integrations.langchain_integration import HashformersTransformer
        transformer = HashformersTransformer(
            segmenter_model="distilgpt2",
            segmenter_type="gpt2",
            device="cpu",
            extract_hashtags=True,
            extract_urls=True
        )
        assert transformer.segmenter_model == "distilgpt2"
        assert transformer.device == "cpu"
        assert transformer.extract_hashtags is True
        assert transformer.extract_urls is True

    def test_transformer_lazy_segmenter(self):
        """HashformersTransformer should lazy-load segmenter."""
        from hashformers.integrations.langchain_integration import HashformersTransformer
        transformer = HashformersTransformer()
        # _segmenter should be None before first access
        assert transformer._segmenter is None

    def test_langchain_available_flag(self):
        """LANGCHAIN_AVAILABLE flag should be boolean."""
        from hashformers.integrations.langchain_integration import LANGCHAIN_AVAILABLE
        assert isinstance(LANGCHAIN_AVAILABLE, bool)

    def test_hashtag_pattern_compiled(self):
        """Hashtag pattern should be pre-compiled."""
        from hashformers.integrations.langchain_integration import HashformersTransformer
        import re
        transformer = HashformersTransformer()
        assert isinstance(transformer._hashtag_pattern, re.Pattern)

    def test_url_slug_pattern_compiled(self):
        """URL slug pattern should be pre-compiled."""
        from hashformers.integrations.langchain_integration import HashformersTransformer
        import re
        transformer = HashformersTransformer()
        assert isinstance(transformer._url_slug_pattern, re.Pattern)


class TestSpacyIntegration:
    """Test the spaCy integration module."""

    def test_import_module(self):
        """spacy_component module should be importable."""
        from hashformers.integrations import spacy_component
        assert spacy_component is not None

    def test_component_class_exists(self):
        """HashformersComponent class should exist."""
        from hashformers.integrations.spacy_component import HashformersComponent
        assert HashformersComponent is not None

    def test_factory_function_exists(self):
        """create_hashformers_component function should exist."""
        from hashformers.integrations.spacy_component import create_hashformers_component
        assert create_hashformers_component is not None

    def test_spacy_available_flag(self):
        """SPACY_AVAILABLE flag should be boolean."""
        from hashformers.integrations.spacy_component import SPACY_AVAILABLE
        assert isinstance(SPACY_AVAILABLE, bool)

    def test_component_registers_with_spacy(self):
        """Component should register with spaCy when available."""
        spacy = pytest.importorskip("spacy", reason="spaCy not installed")
        # Import the module to register the factory
        from hashformers.integrations import spacy_component
        assert "hashformers" in spacy.Language.factories


class TestSpacyComponentInit:
    """Test HashformersComponent initialization."""

    def test_component_init_basic(self):
        """HashformersComponent should initialize with basic params."""
        from hashformers.integrations.spacy_component import HashformersComponent
        component = HashformersComponent(
            nlp=None,
            name="hashformers"
        )
        assert component.name == "hashformers"
        assert component.segmenter_model == "gpt2"

    def test_component_init_custom(self):
        """HashformersComponent should accept custom parameters."""
        from hashformers.integrations.spacy_component import HashformersComponent
        component = HashformersComponent(
            nlp=None,
            name="test",
            segmenter_model="distilgpt2",
            segmenter_type="gpt2",
            device="cpu",
            segment_hashtags=False,
            segment_identifiers=True
        )
        assert component.segmenter_model == "distilgpt2"
        assert component.device == "cpu"
        assert component.segment_hashtags is False
        assert component.segment_identifiers is True

    def test_component_lazy_segmenter(self):
        """HashformersComponent should lazy-load segmenter."""
        from hashformers.integrations.spacy_component import HashformersComponent
        component = HashformersComponent(nlp=None, name="test")
        # _segmenter should be None before first access
        assert component._segmenter is None

    def test_component_patterns_compiled(self):
        """Component patterns should be pre-compiled."""
        from hashformers.integrations.spacy_component import HashformersComponent
        import re
        component = HashformersComponent(nlp=None, name="test")
        assert isinstance(component._hashtag_pattern, re.Pattern)
        assert isinstance(component._camel_case_pattern, re.Pattern)
        assert isinstance(component._identifier_pattern, re.Pattern)


class TestIntegrationsPackage:
    """Test the integrations package structure."""

    def test_import_package(self):
        """integrations package should be importable."""
        from hashformers import integrations
        assert integrations is not None

    def test_langchain_module_accessible(self):
        """langchain_integration should be accessible from integrations."""
        from hashformers.integrations import langchain_integration
        assert hasattr(langchain_integration, 'HashformersTransformer')

    def test_spacy_module_accessible(self):
        """spacy_component should be accessible from integrations."""
        from hashformers.integrations import spacy_component
        assert hasattr(spacy_component, 'HashformersComponent')


class TestStubClasses:
    """Test stub classes when dependencies are not available."""

    def test_document_stub_when_no_langchain(self):
        """Document stub should work when LangChain not installed."""
        from hashformers.integrations.langchain_integration import LANGCHAIN_AVAILABLE
        if not LANGCHAIN_AVAILABLE:
            from hashformers.integrations.langchain_integration import Document
            doc = Document(page_content="test", metadata={"key": "value"})
            assert doc.page_content == "test"
            assert doc.metadata == {"key": "value"}

    def test_document_stub_default_metadata(self):
        """Document stub should default metadata to empty dict."""
        from hashformers.integrations.langchain_integration import LANGCHAIN_AVAILABLE
        if not LANGCHAIN_AVAILABLE:
            from hashformers.integrations.langchain_integration import Document
            doc = Document(page_content="test")
            assert doc.metadata == {}


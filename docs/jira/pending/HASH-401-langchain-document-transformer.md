# HASH-401: Create LangChain Document Transformer

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Story                                                     |
| **Priority**| Medium                                                    |
| **Epic**    | Ecosystem Integrations                                    |

## Goal

Allow LangChain users to drop `hashformers` into a chain for seamless integration with LangChain-based applications.

## Description

Create a LangChain integration that enables users to use hashformers as a document transformer within LangChain pipelines. This will allow word segmentation to be used as a preprocessing step in RAG pipelines, text processing chains, and other LangChain workflows.

## Tasks

1. Create a class inheriting from `BaseDocumentTransformer`.
2. Implement the `transform_documents` method to apply word segmentation to document content.
3. Handle configuration options (model selection, batch processing, etc.).
4. Add proper error handling and logging.
5. Write unit tests for the integration.
6. Document usage examples.

## Example Usage

```python
from langchain.document_transformers import HashformersTransformer
from langchain.schema import Document

transformer = HashformersTransformer(
    segmenter_model="gpt2",
    segmenter_type="incremental"
)

docs = [Document(page_content="#weneedanationalpark is trending")]
transformed = transformer.transform_documents(docs)
# [Document(page_content="we need a national park is trending")]
```

## Deliverable

- A `langchain_integration.py` file in `src/hashformers/integrations/`
- OR a PR to `langchain-community` repository

## Acceptance Criteria

- [ ] `HashformersTransformer` class inherits from `BaseDocumentTransformer`
- [ ] `transform_documents` method correctly segments text in documents
- [ ] Integration works with LangChain's async methods if applicable
- [ ] Unit tests pass
- [ ] Documentation includes usage examples

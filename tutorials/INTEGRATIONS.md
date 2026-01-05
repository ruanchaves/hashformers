# Framework Integrations

Hashformers provides native integrations with popular data processing and NLP frameworks, enabling word segmentation at scale in production environments.

| Framework | Module | Use Case |
|-----------|--------|----------|
| **PySpark** | `hashformers.integrations.spark` | Distributed processing on Databricks/AWS EMR |
| **Haystack** | `hashformers.integrations.haystack` | Document pipelines for search & RAG |
| **LangChain** | `hashformers.integrations.langchain_integration` | LLM chains and document transformers |
| **spaCy** | `hashformers.integrations.spacy_component` | NLP pipeline preprocessing |

---

## PySpark Integration

Process millions of records with hashformers on Databricks, AWS EMR, or any Spark cluster.

### Installation

```bash
pip install hashformers pyspark
```

### Quick Start

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from hashformers.integrations.spark import SparkHashformer

# Initialize Spark
spark = SparkSession.builder \
    .appName("HashformersExample") \
    .getOrCreate()

# Create sample DataFrame
df = spark.createDataFrame([
    ("weneedanationalpark",),
    ("machinelearning",),
    ("AbstractFactoryPattern",),
], ["text"])

# Configure the segmenter
segmenter = SparkHashformer(
    inputCol="text",
    outputCol="segmented",
    segmenterModel="gpt2",
    segmenterType="incremental"
)

# Run segmentation
result = segmenter.transform(df)
result.show(truncate=False)

# +----------------------+----------------------------+
# |text                  |segmented                   |
# +----------------------+----------------------------+
# |weneedanationalpark   |we need a national park     |
# |machinelearning       |machine learning            |
# |AbstractFactoryPattern|Abstract Factory Pattern    |
# +----------------------+----------------------------+
```

### Using in a Spark ML Pipeline

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer

# Chain with other Spark ML transformers
segmenter = SparkHashformer(
    inputCol="hashtag",
    outputCol="segmented_text",
    segmenterModel="gpt2"
)

tokenizer = Tokenizer(
    inputCol="segmented_text",
    outputCol="tokens"
)

pipeline = Pipeline(stages=[segmenter, tokenizer])
model = pipeline.fit(df)
result = model.transform(df)
```

### With Reranker for Higher Accuracy

```python
segmenter = SparkHashformer(
    inputCol="text",
    outputCol="segmented",
    segmenterModel="gpt2",
    segmenterType="incremental",
    rerankerModel="bert-base-uncased",
    rerankerType="masked",
    device="cuda"  # Use GPU if available
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputCol` | `str` | `"text"` | Input column containing text to segment |
| `outputCol` | `str` | `"segmented"` | Output column for segmented text |
| `segmenterModel` | `str` | `"gpt2"` | HuggingFace model name or path |
| `segmenterType` | `str` | `"incremental"` | Model type: `"incremental"`, `"masked"`, or `"seq2seq"` |
| `rerankerModel` | `str` | `None` | Optional reranker model name or path |
| `rerankerType` | `str` | `None` | Reranker model type |
| `device` | `str` | `"cuda"` | Device: `"cuda"` or `"cpu"` |

### Databricks Example

```python
# Databricks notebook cell
from hashformers.integrations.spark import SparkHashformer

# Read from Delta table
df = spark.read.table("social_media.hashtags")

segmenter = SparkHashformer(
    inputCol="hashtag",
    outputCol="segmented_hashtag",
    segmenterModel="gpt2"
)

# Process and write back
result = segmenter.transform(df)
result.write.mode("overwrite").saveAsTable("social_media.hashtags_segmented")
```

### Performance Tips

1. **GPU Acceleration**: Set `device="cuda"` when running on GPU-enabled clusters.
2. **Model Caching**: The model is initialized once per partition to minimize overhead.
3. **Batch Size**: For very large datasets, consider repartitioning to balance load across workers.

---

## Haystack Integration

Integrate hashformers into Haystack 2.0 document processing pipelines for search, RAG, and question answering applications.

### Installation

```bash
pip install hashformers haystack-ai
```

### Quick Start

```python
from haystack import Pipeline, Document
from hashformers.integrations.haystack import HashformersSegmenter

# Create the segmenter component
segmenter = HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=True
)

# Process documents directly
docs = [
    Document(content="Check out #MakeAmericaGreatAgain and #ThrowbackThursday"),
    Document(content="Trending: #weneedanationalpark")
]

result = segmenter.run(documents=docs)

for doc in result["documents"]:
    print(doc.content)
# Check out Make America Great Again and Throwback Thursday
# Trending: we need a national park
```

### Building a Pipeline

```python
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from hashformers.integrations.haystack import HashformersSegmenter

# Create pipeline
pipeline = Pipeline()

# Add components
pipeline.add_component("converter", TextFileToDocument())
pipeline.add_component("segmenter", HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=True,
    segment_camel_case=True
))
pipeline.add_component("cleaner", DocumentCleaner())
pipeline.add_component("splitter", DocumentSplitter(split_by="sentence"))

# Connect components
pipeline.connect("converter", "segmenter")
pipeline.connect("segmenter", "cleaner")
pipeline.connect("cleaner", "splitter")

# Run the pipeline
result = pipeline.run({"converter": {"sources": ["tweets.txt"]}})
```

### Segmentation Options

The `HashformersSegmenter` can target different text patterns:

```python
# Segment only hashtags (default)
segmenter = HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=True,
    segment_urls=False,
    segment_camel_case=False
)

# Segment URL slugs
segmenter = HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=False,
    segment_urls=True  # e.g., "forgot-password-reset" → "forgot password reset"
)

# Segment code identifiers
segmenter = HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=False,
    segment_camel_case=True  # e.g., "getUserById" → "get user by id"
)

# Segment everything
segmenter = HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=True,
    segment_urls=True,
    segment_camel_case=True
)
```

### With Reranker

```python
segmenter = HashformersSegmenter(
    segmenter_model="gpt2",
    segmenter_type="incremental",
    reranker_model="bert-base-uncased",
    reranker_type="masked",
    segment_hashtags=True
)
```

### Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `segmenter_model` | `str` | `"gpt2"` | HuggingFace model name or path |
| `segmenter_type` | `str` | `"incremental"` | Model type: `"incremental"`, `"masked"`, or `"seq2seq"` |
| `reranker_model` | `str` | `None` | Optional reranker model name or path |
| `reranker_type` | `str` | `None` | Reranker model type |
| `device` | `str` | `"cuda"` | Device: `"cuda"` or `"cpu"` |
| `segment_hashtags` | `bool` | `True` | Segment hashtags (e.g., `#MakeAmericaGreatAgain`) |
| `segment_urls` | `bool` | `False` | Segment URL slugs (e.g., `forgot-password-reset`) |
| `segment_camel_case` | `bool` | `False` | Segment camelCase identifiers |

### RAG Pipeline Example

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from hashformers.integrations.haystack import HashformersSegmenter

# Initialize document store
document_store = InMemoryDocumentStore()

# Build indexing pipeline with segmentation
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("segmenter", HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=True
))
indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

indexing_pipeline.connect("segmenter", "embedder")
indexing_pipeline.connect("embedder", "writer")

# Index documents with segmented hashtags
docs = [Document(content="Users are discussing #climatechange and #renewableenergy")]
indexing_pipeline.run({"segmenter": {"documents": docs}})
```

### Privacy-Focused Deployment

Hashformers runs entirely offline—no API calls to external services. This makes it ideal for:

- **GDPR-compliant deployments** in the European market
- **Air-gapped environments** with no internet access
- **Sensitive data processing** where data cannot leave your infrastructure

```python
# All processing happens locally
segmenter = HashformersSegmenter(
    segmenter_model="gpt2",  # Model downloaded once, runs locally
    device="cpu"  # Works without GPU
)
```

---

## See Also

- [Basic Usage](../README.md#basic-usage) — Getting started with hashformers
- [Model Types](../README.md#what-models-can-i-use) — Understanding incremental, masked, and seq2seq models
- [Evaluation Guide](./EVALUATION.md) — Benchmarking segmentation quality


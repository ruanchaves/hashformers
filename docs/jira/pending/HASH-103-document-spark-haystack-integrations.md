# HASH-103: Document PySpark and Haystack Integrations

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Story                                                     |
| **Priority**| Medium                                                    |
| **Epic**    | Epic 1: Packaging & Positioning                           |
| **Files**   | `README.md`, `tutorials/INTEGRATIONS.md` (new)            |

## Description

The project source tree includes integration modules for PySpark (`src/hashformers/integrations/spark.py`) and Haystack (`src/hashformers/integrations/haystack.py`).

While HASH-401 and HASH-402 track the documentation/implementation for LangChain and spaCy, the Spark and Haystack integrations are currently orphaned: they exist in the codebase but are completely undocumented in the `README.md` or `tutorials/`.

Documenting the Spark integration is particularly important to support the **"Enterprise use-cases"** and **"Offline/Infrastructure"** goals established in HASH-102.

### Current State

- The `README.md` lists "Applications" but provides no code examples for using hashformers in a distributed environment (Spark) or within a search pipeline (Haystack).
- Both integrations are fully implemented but invisible to users:
  - `SparkHashformer` — a PySpark ML Transformer for Databricks/AWS EMR
  - `HashformersSegmenter` — a Haystack 2.0 component for document pipelines

## Tasks

1. **Create Integration Guide:** Create a new markdown file (`tutorials/INTEGRATIONS.md`) consolidating all integration documentation.

2. **Document PySpark Integration:**
   - Explain how to use `SparkHashformer` in a Spark ML Pipeline.
   - Provide a code snippet for applying segmentation to a DataFrame column.
   - Document the available parameters (`inputCol`, `outputCol`, `segmenterModel`, `segmenterType`, `rerankerModel`, `device`).
   - Mention Databricks/AWS EMR compatibility.

3. **Document Haystack Integration:**
   - Explain how to use `HashformersSegmenter` in a Haystack 2.0 pipeline.
   - Provide code examples showing hashtag, URL, and camelCase segmentation options.
   - Highlight privacy-focused European deployment use cases.

4. **Update README:** Add an "Integrations" or "Advanced Usage" section in `README.md` linking to the integration guide.

## Example Documentation Snippets

### PySpark Example

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from hashformers.integrations.spark import SparkHashformer

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame([
    ("weneedanationalpark",),
    ("machinelearning",)
], ["hashtag"])

segmenter = SparkHashformer(
    inputCol="hashtag",
    outputCol="segmented",
    segmenterModel="gpt2"
)

pipeline = Pipeline(stages=[segmenter])
model = pipeline.fit(df)
result = model.transform(df)
result.show()
```

### Haystack Example

```python
from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from hashformers.integrations.haystack import HashformersSegmenter

segmenter = HashformersSegmenter(
    segmenter_model="gpt2",
    segment_hashtags=True,
    segment_camel_case=True
)

pipeline = Pipeline()
pipeline.add_component("converter", TextFileToDocument())
pipeline.add_component("segmenter", segmenter)
pipeline.connect("converter", "segmenter")

result = pipeline.run({"converter": {"sources": ["tweets.txt"]}})
```

## Acceptance Criteria

- [ ] A clear code example demonstrating how to run hashformers on a Spark DataFrame is available in `tutorials/INTEGRATIONS.md`.
- [ ] A code example showing hashformers integration with Haystack 2.0 is available.
- [ ] Installation instructions for optional dependencies (`pyspark`, `haystack-ai`) are documented.
- [ ] The `README.md` is updated to mention support for these frameworks, reinforcing the library's versatility.
- [ ] Parameter reference tables are included for both integrations.

## Related Resources

- `src/hashformers/integrations/spark.py` — PySpark ML Transformer implementation
- `src/hashformers/integrations/haystack.py` — Haystack 2.0 component implementation
- HASH-102 — README rebranding (establishes enterprise positioning)
- HASH-401 — LangChain integration documentation
- HASH-402 — spaCy integration documentation


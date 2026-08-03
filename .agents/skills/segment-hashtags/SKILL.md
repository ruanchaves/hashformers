---
name: segment-hashtags
description: Segment hashtags or tweets, sample and process hashtag files without copying their contents into agent context, discover and configure language-appropriate public Hub models, apply regex rules, and rank precomputed segmentation candidates with the Hashformers MCP server. Use when a user asks to split or decompound hashtags, run bulk local segmentation, select a model for an unknown language, replace hashtags in tweets, compare candidates, rerank hypotheses, or inspect component scores.
---

# Segment with Hashformers

Use the configured Hashformers MCP server. Never reproduce the algorithm, load a
model directly, or guess segmentations when the tools are unavailable.

## Workflow

1. Confirm that the required Hashformers MCP tool is available. If it is not,
   explain that `hashformers[mcp]` must be installed and `hashformers-mcp`
   configured, then stop.
2. Determine the language before Transformer segmentation. When the user did
   not supply it:
   - For inline inputs, inspect only a small sample already present in context;
     never call a remote service with the hashtags.
   - For a file, call `sample_hashtag_file` and use only its at-most-20 distinct
     samples. Never open the whole file or copy it into tool arguments.
   - State the inferred dominant language and confidence. If the sample is
     ambiguous or mixed, ask the user. Prefer a multilingual model without
     asking only when it is a clearly safe fallback for the request.
3. If the server reports that model selection is deferred and unconfigured,
   follow the one-time model-selection workflow below before segmentation.
4. Choose one segmentation tool:
   - Use `start_hashtag_file_job`, then `continue_hashtag_file_job`, when the
     user identifies a local input file.
   - Use `segment_tweets` for complete tweets whose hashtags must be replaced.
   - Use `segment_with_regex` when the user requests deterministic regex rules.
   - Use `rank_candidates` for hypotheses and scores already supplied by the
     user or another tool.
   - Otherwise use `segment_hashtags`.
5. Preserve input spelling, order, and duplicates. Put up to 64 interactive
   hashtags in one call instead of calling once per hashtag. Split larger
   inline requests into ordered batches; use file jobs when the input is a file.
6. Include nondefault options only when the user requests them or they are
   necessary for the task.
7. Report selected segmentations. Include candidates or component rankings only
   when requested.

## Deferred Model Selection

Use this workflow only when `deferred_model_selection` is true and
`models_configured` is false in a sampling or discovery response. That state
means the operator started the server with `--defer-model-selection`, which is
authorization for one validated public Hub selection and its later download.
Proceed directly without asking for a second download confirmation.

1. Call `discover_huggingface_models` with the inferred language tag (for
   example `en` or `pt`), `role="segmenter"`, and the default bounded limit.
2. Choose a candidate based on language and architecture compatibility. Treat
   tags, likes, and downloads only as shortlist signals, never as proof of the
   universally best segmenter.
3. Discover a reranker only when the user needs reranker or ensemble selection.
   The official Hugging Face MCP may be used for broader exploration when it is
   already available, but never make it a prerequisite. In either case, pass
   the selected repository and exact revision to Hashformers so it can re-fetch
   and validate the model itself.
4. Call `configure_models` with the segmenter repository ID, exact revision,
   and returned scorer type, plus the same three fields for an optional
   reranker. Do not substitute a branch name such as `main` for the revision.
5. Before the first segmentation or file-job continuation, report the selected
   repository IDs and exact revisions. That next inference may perform the
   potentially large pinned download.

An identical `configure_models` retry is idempotent. If another selection is
already configured, explain that changing it requires an MCP server restart;
never attempt to hot-swap or retain multiple models. Failed discovery or
validation is not permission to use an unvalidated repository.

## Large Files

Pass file paths to `start_hashtag_file_job`; never read the whole file and copy
its hashtags into a tool argument. The server indexes text, CSV, or JSON Lines,
deduplicates normalized hashtags, and creates a persistent local checkpoint.

When language is unknown, call `sample_hashtag_file` first with the same
`input_format` and `input_field` that the job will use. Its default and maximum
sample size is 20. It streams the file with bounded memory and returns only a
distinct reservoir plus compact counts and format metadata. Never pass the
sample or any other file content to Hugging Face discovery.

Pass the returned `job_path` to `continue_hashtag_file_job`. Keep calling it
until `status` is `completed`; each call processes a bounded number of unique
hashtags and can be retried after a timeout or client restart. If the user asks
to stop, make no further continuation calls. No work runs in the background.

Use `input_field` when a CSV column or JSON object field is not named `hashtag`.
Let the server derive an output path unless the user specifies one. Set
`overwrite=true` only when the user explicitly authorizes replacing the target
and the server was started with `--allow-file-overwrite`. File paths must be
inside a directory authorized by the server's `--file-root` startup option.

After the call, report the output path and summary counts. Do not open or print
the complete output file unless the user explicitly asks; doing so would place
the dataset back into agent context. If verification is requested, inspect only
the requested rows or a small sample.

## Transformer Options

For `segment_hashtags`, Transformer-mode `segment_tweets`, and file jobs:

- `top_k` is the beam-search width and can change the selected segmentation.
- `steps` is the beam-search depth. MCP calls cap `top_k` at 64 and `steps` at
  32 to keep requests bounded. The server can also reject a batch whose combined
  input lengths, beam width, and depth exceed its aggregate work budget; retry a
  smaller batch or reduce `top_k` or `steps`.
- `max_candidates` only limits serialized alternatives; it does not narrow the
  search. It must be between 1 and 64.
- `ranking_strategy` can be `auto`, `segmenter`, `reranker`, or `ensemble`.
  Reranker and ensemble strategies require a reranker configured when the MCP
  server starts.
- `alpha` and `beta` configure ensemble selection.
- `lower`, `remove_hashtag`, and `hashtag_character` configure preprocessing.
- `include_component_rankings=true` returns segmenter, reranker, and ensemble
  rankings that actually ran.

Model names, scorer types, devices, and model batch sizes are normally
server-startup settings. The only exception is one exact-revision selection
authorized by `--defer-model-selection`. If an ordinarily configured model is
unsuitable, tell the user which `hashformers-mcp` startup option must change
rather than calling `configure_models` or silently choosing another model.

## Tool Contracts

Call `segment_hashtags` with a `hashtags` list and optional Transformer options.
Expect a `results` list preserving input order. Each item contains the original
and normalized input, selected segmentation, selected ranking strategy, ordered
lower-is-better candidates, and optional component rankings. The response also
records the selected repository IDs and revisions.

Call `sample_hashtag_file` with `input_path` and optional `input_format`,
`input_field`, and `sample_size`. Expect no more than 20 distinct samples and
compact local metadata, never the full dataset.

Call `discover_huggingface_models` with a language tag and `segmenter` or
`reranker` role. It returns at most the requested hard-capped number of public,
non-gated, size-bounded candidates with scorer type, architecture, language
tags, size metadata, exact revision, and match reason.

Call `configure_models` only in authorized deferred mode. Supply exact revision
SHAs for the selected segmenter and optional reranker. Configuration validates
metadata without loading a model; loading remains lazy until inference.

Call `start_hashtag_file_job` with `input_path` and optional `output_path`,
`input_format`, `input_field`, `overwrite`, and Transformer options. It indexes
the file without model inference and returns a persistent `job_path`.

Call `continue_hashtag_file_job` with that path and optional
`max_unique_hashtags`. Expect only paths, checkpoint counts, and active model
metadata including exact revisions. Repeat while status is `in_progress`;
segmentation records appear in
the final output file only when the job completes. The maximum value is 64;
call the tool repeatedly instead of requesting a larger chunk.

Call `segment_with_regex` with `inputs` and optional ordered `regex_rules` and
preprocessing options. Rules are applied sequentially to each input.

Call `segment_tweets` with `tweets` and `segmenter_kind` set to `regex` or
`transformer`. It returns transformed text and per-occurrence hashtag results.
Tweet extraction supports the standard `#` marker only.

Call `rank_candidates` with candidate sets shaped as:

```json
{
  "candidate_sets": [
    {
      "input": "#icecold",
      "candidates": [
        {"segmentation": "ice cold", "score": 1.2},
        {"segmentation": "i ce cold", "score": 2.3}
      ]
    }
  ],
  "ranking_strategy": "segmenter"
}
```

Use `segmenter` to select from supplied scores without loading a model. Use
`reranker` or `ensemble` only when the server has a configured reranker.

## Examples

For “Segment `#blacklivesmatter`,” call:

```json
{"hashtags": ["#blacklivesmatter"]}
```

For “Show the top three returned candidates using a beam width of 20,” call:

```json
{
  "hashtags": ["#therapist"],
  "top_k": 20,
  "max_candidates": 3
}
```

For “Segment every hashtag in `data/hashtags.csv` without loading it into the
chat,” call `start_hashtag_file_job`:

```json
{
  "input_path": "data/hashtags.csv",
  "input_field": "hashtag"
}
```

Then call `continue_hashtag_file_job` with the returned checkpoint until it
completes:

```json
{
  "job_path": "/absolute/path/hashtags.jsonl.job.sqlite3",
  "max_unique_hashtags": 64
}
```

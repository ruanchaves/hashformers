---
name: segment-hashtags
description: Segment hashtags with the Hashformers MCP tool. Use when a user asks to split or decompound one or more hashtags, obtain the best segmentation, or compare ranked segmentation candidates and their scores.
---

# Segment Hashtags

Use the configured Hashformers MCP server for segmentation. Never reproduce the
segmentation algorithm, load a model directly, or guess a segmentation when the
tool is unavailable.

## Workflow

1. Confirm that the Hashformers MCP tool `segment_hashtags` is available. If it
   is unavailable, explain that the local Hashformers MCP server must be
   installed and configured, then stop.
2. Pass all requested hashtags together in the `hashtags` array, preserving
   their spelling, order, and duplicates.
3. Include `top_k` only when the user requests a particular number of ranked
   candidates. It must be an integer greater than zero.
4. Call `segment_hashtags` once. Do not invoke package code or implement a
   fallback segmentation.
5. Report each `selected_segmentation`. Include the ordered `candidates` and
   their `score` values when the user asks for alternatives or ranking details.

## Tool Contract

Call `segment_hashtags` with:

- `hashtags`: a list of hashtag strings.
- `top_k`: an optional positive integer limiting candidates per hashtag. It
  defaults to 5.

Expect a JSON-serializable object whose `results` list preserves input order:

```json
{
  "results": [
    {
      "input": "#examplehashtag",
      "selected_segmentation": "example hashtag",
      "candidates": [
        {"segmentation": "example hashtag", "score": 1.23}
      ]
    }
  ]
}
```

Candidates are ordered best first; lower scores are better. Do not reorder them
or compare scores across different hashtags.

## Examples

For “Segment `#blacklivesmatter`,” call:

```json
{"hashtags": ["#blacklivesmatter"]}
```

For “Show the top three candidates for `#therapist`,” call:

```json
{"hashtags": ["#therapist"], "top_k": 3}
```

# August 3, 2026 Colab T4 FP16 result

This directory contains the complete fixed-protocol artifacts for the prompted
Qwen comparison described in the parent [benchmark report](../../README.md).
Both runs used the committed 280-record manifest and repository revision
`b30e66e163bb5ac9d43da23edd725eda7353adf3`; each metadata file reports
`status: completed`, zero runtime errors, and `repository_dirty: false`.

Files:

- `qwen3/predictions.jsonl` and `qwen3/run_metadata.json`: pinned
  Qwen3-0.6B, text-only, non-thinking, unquantized FP16;
- `qwen2/predictions.jsonl` and `qwen2/run_metadata.json`: pinned
  Qwen2-0.5B-Instruct under the same refreshed zero-shot FP16 protocol;
- `comparison.json`: per-run summaries and the paired 10,000-resample
  bootstrap accuracy-difference interval.

SHA-256 checksums:

```text
6347bf2ec1a9638d28d8f4c0d67758cf67ac466ff66f63cf6821b9d69dca9b27  qwen3/predictions.jsonl
d8ad8939f24ff340b06e9b8a7c2191403c8a6e436de9aaae05e65be0baab9d98  qwen3/run_metadata.json
2215656898e80d2d50bea045022f93a9ebef34a10798a2381ac09764f97082d1  qwen2/predictions.jsonl
e4528fedb490f21e21c8139e98eabdbad1d0ee0f1afff11781be7ca6d14f8284  qwen2/run_metadata.json
8278089cef9664e65c94ce3df4440b93a29c62764f10be54dff4ae071d6a327b  comparison.json
```

The exact-match accuracy difference, Qwen3 minus Qwen2, is 2.14 percentage
points with a 95% paired bootstrap interval of −1.43 to 5.71 points. This does
not establish an accuracy difference and must not be generalized to other
prompted models or compared with historical Hashformers rows that were not run
on this manifest.

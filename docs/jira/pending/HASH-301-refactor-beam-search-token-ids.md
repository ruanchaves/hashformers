# HASH-301: Refactor Beam Search to operate on Token IDs instead of Strings

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Story                                                     |
| **Priority**| High                                                      |
| **Epic**    | Epic 3: Performance Optimization (The "Career Maker")     |
| **Files**   | `src/hashformers/beamsearch/algorithm.py`                 |

## Description

Currently, `algorithm.py` generates string candidates (`"word1"`, `"word1 word2"`) and re-tokenizes them every step. This is inefficient and creates an O(N²) complexity bottleneck.

### Current Flow (Inefficient)

```
run() loop iteration:
    1. next_step() generates string candidates by inserting space characters
    2. update_probabilities() calls model.get_probs(strings)
    3. get_probs() tokenizes each string ← REDUNDANT WORK
    4. Model processes tokens
    5. trim_tree() operates on strings
    
Result: Tokenizer is called O(candidates × steps) times
```

### Target Flow (Optimized)

```
run() loop iteration:
    1. Tokenize input ONCE at the start
    2. next_step() operates on token ID tensors
    3. update_probabilities() passes token IDs directly to model
    4. Model processes tokens (no re-tokenization)
    5. trim_tree() operates on token IDs, decodes only when necessary
    
Result: Tokenizer is called O(1) times
```

## Current Implementation Analysis

**`next_step()` - String-based candidate generation:**
```python
def next_step(self, list_of_candidates: List[str]) -> List[str]:
    output = []
    for candidate_string in list_of_candidates:
        candidates = (
            candidate_string[:pos] + ' ' + candidate_string[pos:]
            if pos else candidate_string 
            for pos in range(len(candidate_string))
        )
        filtered = [x for x in candidates if not DOUBLE_SPACE_PATTERN.findall(x)]
        output.extend(filtered)
    return output
```

**`run()` - Main loop:**
```python
def run(self, dataset: List[str], topk: int = 20, steps: int = 13):
    tree = dataset
    prob_dict: ProbabilityDict = {}
    for i in range(steps):
        tree = self.next_step(tree)               # String manipulation
        tree = self.reshape_tree(tree, ...)
        prob_dict = self.update_probabilities(tree, prob_dict)  # Re-tokenizes!
        tree = self.flatten_list(tree)
        tree = self.trim_tree(tree, prob_dict, topk)
    return ProbabilityDictionary(prob_dict)
```

## Tasks

1. **Change `next_step` signature to accept `List[torch.Tensor]` (Token IDs) instead of Strings:**
   ```python
   def next_step(self, token_sequences: List[torch.Tensor]) -> List[torch.Tensor]:
       # Instead of inserting space character, insert space token ID
       space_token_id = self.tokenizer.encode(' ', add_special_tokens=False)[0]
       # ... generate new token sequences
   ```

2. **Pre-tokenize input at the start of `run()`:**
   ```python
   def run(self, dataset: List[str], topk: int = 20, steps: int = 13):
       # Tokenize once at the start
       token_sequences = [
           self.tokenizer.encode(text, add_special_tokens=False, return_tensors='pt')
           for text in dataset
       ]
       # ... rest of the loop operates on token_sequences
   ```

3. **Update `update_probabilities()` to pass Token IDs directly to the model:**
   - Remove tokenization from `model.get_probs()`
   - Accept token tensors instead of strings

4. **Update `trim_tree()` to operate on Token IDs:**
   - Decode back to strings only when necessary (e.g., for regex checks or final output)
   - Consider performing double-space checks on Token IDs directly if possible:
     ```python
     # Check for consecutive space tokens instead of regex
     def has_double_space(token_ids: torch.Tensor, space_token_id: int) -> bool:
         for i in range(len(token_ids) - 1):
             if token_ids[i] == space_token_id and token_ids[i+1] == space_token_id:
                 return True
         return False
     ```

5. **Update `ProbabilityDict` type alias:**
   ```python
   # Old: Dict[str, float]
   # New: Dict[Tuple[int, ...], float]  or Dict[bytes, float] for hashable keys
   TokenSequenceKey = Tuple[int, ...]
   ProbabilityDict = Dict[TokenSequenceKey, Score]
   ```

> [!IMPORTANT]
> This is a foundational change required for HASH-302 (KV-Caching). Without Token ID-based operations, we cannot implement efficient caching.

## Technical Considerations

- **Hashable keys:** `torch.Tensor` is not hashable. Use `tuple(tensor.tolist())` or `tensor.tobytes()` as dictionary keys.
- **Space token handling:** Different tokenizers represent spaces differently (e.g., GPT-2 uses `Ġ` prefix). Handle this carefully.
- **Batch dimension:** Decide whether to maintain batch dimension in tensors or use lists of 1D tensors.

## Acceptance Criteria

- [ ] The main loop in `run()` operates primarily on Tensors.
- [ ] Tokenizer is called *once* at the start, not inside the loop.
- [ ] All existing unit tests pass (output should be identical).
- [ ] Benchmark shows measurable speedup on 50+ character inputs.
- [ ] Code is ready for HASH-302 KV-caching integration.

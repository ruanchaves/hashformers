# HASH-302: Implement KV-Caching for Incremental Segmentation

| Field         | Value                                                   |
|---------------|---------------------------------------------------------|
| **Type**      | Story                                                   |
| **Priority**  | High                                                    |
| **Epic**      | Epic 3: Performance Optimization (The "Career Maker")   |
| **Dependencies** | HASH-201, HASH-301                                   |
| **Files**     | `src/hashformers/beamsearch/gpt2_lm.py`, `src/hashformers/beamsearch/algorithm.py` |

## Description

To achieve O(N) complexity instead of O(N²), we must cache the Transformer's key-value attention states. When evaluating a split at index `i`, we should not re-compute attention layers for indices `0..i-1`.

### Current Complexity Analysis

For a string of length N with beam width K:
- **Current:** Each step re-processes the entire sequence → O(N² × K) forward passes
- **With KV-Cache:** Each step only processes new tokens → O(N × K) forward passes

### How KV-Caching Works

```
Step 1: Process "hello"
    → Compute attention for all tokens
    → Cache key/value tensors: past_key_values_1

Step 2: Process "hello world" 
    Current: Recompute attention for "hello" + "world"
    Optimized: Pass past_key_values_1, only compute for "world"
    → Returns updated cache: past_key_values_2
```

## Tasks

### 1. Update `GPT2LM` to accept and return `past_key_values`

```python
class GPT2LM:
    def get_probs_with_cache(
        self, 
        new_token_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None
    ) -> Tuple[List[float], Tuple]:
        """
        Score sequences using cached attention states.
        
        Args:
            new_token_ids: Only the NEW tokens to process
            past_key_values: Cached attention states from previous computation
            
        Returns:
            probs: Probability scores for each sequence
            new_past_key_values: Updated cache including new tokens
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_token_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
        # Extract probabilities...
        
        return probs, outputs.past_key_values
```

### 2. Modify `Node` structure in `data_structures.py` to store cache

```python
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Node:
    hypothesis: str
    characters: str
    score: float
    token_ids: Optional[Tuple[int, ...]] = None
    past_key_values: Optional[Tuple] = None  # KV cache for this node
```

### 3. Update `algorithm.py` to propagate cache through beam search

```python
def run(self, dataset: List[str], topk: int = 20, steps: int = 13):
    # Initialize nodes with empty cache
    nodes = [Node(
        hypothesis=text,
        characters=text.replace(" ", ""),
        score=0.0,
        token_ids=self.tokenizer.encode(text),
        past_key_values=None
    ) for text in dataset]
    
    for i in range(steps):
        # Generate children - each child inherits parent's cache
        children = self.next_step_with_cache(nodes)
        
        # Score children - only process NEW tokens
        for child in children:
            parent = child.parent
            new_tokens = child.token_ids[len(parent.token_ids):]
            
            prob, new_cache = self.model.get_probs_with_cache(
                new_tokens,
                parent.past_key_values
            )
            
            child.score = prob
            child.past_key_values = new_cache
        
        # Prune to top-k
        nodes = self.trim_tree_nodes(children, topk)
    
    return nodes
```

### 4. Implement cache memory management

> [!WARNING]
> KV cache can grow large, especially with many beam candidates. Implement limits.

```python
class CacheManager:
    def __init__(self, max_cached_nodes: int = 100):
        self.max_cached_nodes = max_cached_nodes
        self.cache_size_bytes = 0
        self.max_cache_bytes = 2 * 1024**3  # 2GB limit
    
    def should_evict(self) -> bool:
        return self.cache_size_bytes > self.max_cache_bytes
    
    def evict_low_score_caches(self, nodes: List[Node]) -> None:
        """Clear caches for low-scoring nodes that are unlikely to win."""
        if not self.should_evict():
            return
        
        sorted_nodes = sorted(nodes, key=lambda n: n.score, reverse=True)
        for node in sorted_nodes[self.max_cached_nodes:]:
            node.past_key_values = None  # Free memory
```

### 5. Handle cache shape mismatches for batched inference

Different sequence lengths in a batch require padding/masking of the KV cache:

```python
def batch_with_cache(self, nodes: List[Node]) -> Tuple[torch.Tensor, Tuple]:
    # Pad caches to same length for batched inference
    max_cache_len = max(self._cache_length(n.past_key_values) for n in nodes)
    
    padded_caches = [
        self._pad_cache(n.past_key_values, max_cache_len) 
        for n in nodes
    ]
    
    # Stack into batch
    batched_cache = self._stack_caches(padded_caches)
    return batched_cache
```

## Performance Expectations

| Input Length | Current Time | With KV-Cache | Speedup |
|--------------|--------------|---------------|---------|
| 20 chars     | 100ms        | 50ms          | 2x      |
| 50 chars     | 500ms        | 100ms         | 5x      |
| 100 chars    | 2000ms       | 200ms         | 10x     |

> [!NOTE]
> Actual speedup depends on hardware are beam width. The quadratic-to-linear improvement is most dramatic for longer inputs.

## Acceptance Criteria

- [ ] Inference speed on long strings (>50 chars) improves by at least **10x**.
- [ ] Memory usage is monitored and capped (implement cache eviction).
- [ ] All existing unit tests pass with identical outputs.
- [ ] New benchmark test demonstrates the speedup.
- [ ] Works correctly with batched inference.
- [ ] Graceful degradation when cache memory limit is reached.

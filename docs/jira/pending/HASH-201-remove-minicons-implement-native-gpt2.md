# HASH-201: Remove `minicons` dependency and Implement Native GPT-2 Scoring

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Task                                                      |
| **Priority**| High                                                      |
| **Epic**    | Epic 2: Core Refactor (Remove `minicons`)                 |
| **Files**   | `src/hashformers/beamsearch/gpt2_lm.py`, `src/hashformers/beamsearch/minicons_lm.py` |

## Description

We are removing the `minicons` library to reduce dependency risk and enable access to the model's `past_key_values` for future KV-caching optimization (HASH-302). We must rewrite `GPT2LM` to use Hugging Face `AutoModelForCausalLM` directly.

### Current Architecture

```
GPT2LM (gpt2_lm.py)
    └── inherits from MiniconsLM (minicons_lm.py)
            └── uses minicons.scorer.IncrementalLMScorer
```

**Current `GPT2LM` implementation:**
```python
from hashformers.beamsearch.minicons_lm import MiniconsLM

class GPT2LM(MiniconsLM):
    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
            model_type='IncrementalLMScorer'
        )
```

**Current `MiniconsLM.get_probs()` flow:**
```python
def get_probs(self, list_of_candidates):
    probs = []
    dl = DataLoader(list_of_candidates, batch_size=self.gpu_batch_size)
    with torch.no_grad():
        for batch in dl:
            probs.extend(self.get_batch_scores(batch))
    return probs
```

## Tasks

1. **Modify `GPT2LM` in `gpt2_lm.py`:**
   - Remove inheritance from `MiniconsLM`.
   - Directly load `AutoModelForCausalLM` and `AutoTokenizer` from Hugging Face.

2. **Implement a `get_probs(list_of_candidates)` method using native PyTorch:**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch
   from torch.utils.data import DataLoader
   
   class GPT2LM:
       def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20):
           self.device = device
           self.gpu_batch_size = gpu_batch_size
           self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
           self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
           self.model.eval()
           
           # Handle padding token
           if self.tokenizer.pad_token is None:
               self.tokenizer.pad_token = self.tokenizer.eos_token
       
       def get_probs(self, list_of_candidates):
           probs = []
           dl = DataLoader(list_of_candidates, batch_size=self.gpu_batch_size)
           
           with torch.no_grad():
               for batch in dl:
                   inputs = self.tokenizer(
                       list(batch), 
                       return_tensors='pt', 
                       padding=True,
                       return_attention_mask=True
                   ).to(self.device)
                   
                   outputs = self.model(**inputs, labels=inputs['input_ids'])
                   
                   # Calculate per-sequence log-likelihood
                   # Use outputs.loss or compute from logits
                   batch_probs = self._compute_sequence_scores(outputs, inputs)
                   probs.extend(batch_probs)
           
           return probs
   ```

3. **Implement `_compute_sequence_scores()` helper:**
   - Calculate log-likelihood from `logits` for each sequence.
   - Handle padding tokens correctly (mask them out of the calculation).
   - Return `1 - sum(log_probs)` to match current `MiniconsLM` output format.

4. **Delete `minicons` from `setup.py` and `requirements.txt`.**

5. **Update imports** in any files that reference `MiniconsLM`.

> [!WARNING]
> Ensure you handle the "sliding window" or batching correctly to avoid OOM on large candidate lists. The current implementation uses `gpu_batch_size=1000` in `algorithm.py`.

## Technical Notes

- The current scoring uses `1 - sum(log_probs)` as the score (lower is better).
- Ensure left-padding for causal LM batched inference (GPT-2 standard practice).
- Consider using `torch.nn.CrossEntropyLoss(reduction='none')` for per-token loss computation.

## Acceptance Criteria

- [ ] Unit tests pass without `minicons` installed.
- [ ] Output probabilities match the previous implementation (within float tolerance ε < 1e-5).
- [ ] `GPT2LM` can be instantiated with any `AutoModelForCausalLM`-compatible model (GPT-2, GPT-J, etc.).
- [ ] No `minicons` imports remain in the codebase.
- [ ] `minicons` removed from `setup.py` `install_requires`.

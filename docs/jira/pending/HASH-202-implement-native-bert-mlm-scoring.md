# HASH-202: Implement Native BERT Masked Language Model Scoring

| Field       | Value                                                     |
|-------------|-----------------------------------------------------------|
| **Type**    | Task                                                      |
| **Priority**| Medium                                                    |
| **Epic**    | Epic 2: Core Refactor (Remove `minicons`)                 |
| **Files**   | `src/hashformers/beamsearch/bert_lm.py`                   |

## Description

Similar to HASH-201, we need to replace the `minicons` implementation for BERT with a native Hugging Face implementation using `AutoModelForMaskedLM`.

### Current Architecture

```
BertLM (bert_lm.py)
    └── inherits from MiniconsLM (minicons_lm.py)
            └── uses minicons.scorer.MaskedLMScorer
```

**Current `BertLM` implementation:**
```python
from hashformers.beamsearch.minicons_lm import MiniconsLM

class BertLM(MiniconsLM):
    def __init__(self, model_name_or_path, gpu_batch_size=1, gpu_id=0):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device='cuda',
            gpu_batch_size=gpu_batch_size,
            model_type='MaskedLMScorer'
        )
```

## Tasks

1. **Modify `BertLM` in `bert_lm.py`:**
   - Remove inheritance from `MiniconsLM`.
   - Directly load `AutoModelForMaskedLM` and `AutoTokenizer` from Hugging Face.

2. **Implement Pseudo-Log-Likelihood (PLL) scoring manually:**
   
   For a sentence with tokens [w₁, w₂, ..., wₙ]:
   - Iteratively mask each token position i
   - Compute P(wᵢ | w₁, ..., wᵢ₋₁, [MASK], wᵢ₊₁, ..., wₙ)
   - Sum log-probabilities: PLL = Σ log P(wᵢ | context)

   ```python
   from transformers import AutoModelForMaskedLM, AutoTokenizer
   import torch
   
   class BertLM:
       def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=1):
           self.device = device
           self.gpu_batch_size = gpu_batch_size
           self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
           self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path).to(device)
           self.model.eval()
       
       def get_probs(self, list_of_candidates):
           probs = []
           for candidate in list_of_candidates:
               pll = self._compute_pll(candidate)
               probs.append(pll)
           return probs
       
       def _compute_pll(self, text):
           # Tokenize once
           tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
           input_ids = tokens['input_ids'][0]
           
           # Skip [CLS] and [SEP] tokens
           token_indices = range(1, len(input_ids) - 1)
           
           log_prob_sum = 0.0
           
           for i in token_indices:
               # Create masked version
               masked_ids = input_ids.clone()
               original_token = masked_ids[i].item()
               masked_ids[i] = self.tokenizer.mask_token_id
               
               with torch.no_grad():
                   outputs = self.model(masked_ids.unsqueeze(0))
                   logits = outputs.logits[0, i]
                   log_probs = torch.log_softmax(logits, dim=-1)
                   log_prob_sum += log_probs[original_token].item()
           
           return -log_prob_sum  # Return negative for consistency (lower is better)
   ```

3. **Optimization:** Use `inputs.repeat()` to batch the masked versions of a single sentence to run in one forward pass if VRAM allows:
   
   ```python
   def _compute_pll_batched(self, text):
       tokens = self.tokenizer(text, return_tensors='pt')
       input_ids = tokens['input_ids'][0]
       seq_len = len(input_ids)
       
       # Create all masked versions at once (excluding [CLS] and [SEP])
       num_tokens = seq_len - 2  # Exclude special tokens
       
       # Repeat input for each position to mask
       batched_ids = input_ids.unsqueeze(0).repeat(num_tokens, 1)
       
       # Mask each position
       for i in range(num_tokens):
           batched_ids[i, i + 1] = self.tokenizer.mask_token_id
       
       batched_ids = batched_ids.to(self.device)
       
       with torch.no_grad():
           outputs = self.model(batched_ids)
           logits = outputs.logits
       
       # Extract log-prob for each original token
       log_prob_sum = 0.0
       for i in range(num_tokens):
           pos = i + 1  # Offset for [CLS]
           original_token = input_ids[pos].item()
           log_probs = torch.log_softmax(logits[i, pos], dim=-1)
           log_prob_sum += log_probs[original_token].item()
       
       return -log_prob_sum
   ```

> [!NOTE]
> The batched approach uses O(n²) memory for a sentence of n tokens since we create n copies. For very long sequences, fall back to the sequential approach.

## Acceptance Criteria

- [ ] `BertLM` functions correctly using `AutoModelForMaskedLM`.
- [ ] PLL scoring matches `minicons` implementation (within float tolerance ε < 1e-4).
- [ ] Batched optimization is implemented with graceful fallback for long sequences.
- [ ] Works with any BERT-like model (BERT, RoBERTa, DistilBERT, etc.).
- [ ] Unit tests pass without `minicons` installed.

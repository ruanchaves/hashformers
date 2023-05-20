from lm_scorer.models.auto import GPT2LMScorer
from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers.tokenization_utils import BatchEncoding

class PaddedGPT2LMScorer(GPT2LMScorer):
    """A Language Model (LM) scorer using GPT2 and supporting token padding.

    Inherits from the GPT2LMScorer class to score sentences. Additionally, supports padding of tokens for
    consistent length sequences.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    def _build(self, model_name: str, options: Dict[str, Any]) -> None:
        """Initializes the tokenizer and model with the given model_name and options.

        Args:
            model_name (str): Name of the model to be used for tokenization and generation.
            options (Dict[str, Any]): Additional options for the model, such as 'device'.
        """
        super()._build(model_name, options)

        # pylint: disable=attribute-defined-outside-init
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, add_special_tokens=False
        )
        # Add the pad token to GPT2 dictionary.
        # len(tokenizer) = vocab_size + 1
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
        self.tokenizer.pad_token = "<|pad|>"

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # We need to resize the embedding layer because we added the pad token.
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        if "device" in options:
            self.model.to(options["device"])

    def _tokens_log_prob_for_batch(
        self, text: List[str]
    ) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:
        """Calculates the log probability of tokens for a batch of sentences.

        Args:
            text (List[str]): List of sentences to calculate log probabilities for.

        Returns:
            List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]: List of tuples, each containing the 
            log probabilities, ids, and tokens for a sentence.
        """
        outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []
        if len(text) == 0:
            return outputs

        # TODO: Handle overflowing elements for long sentences
        text = list(map(self._add_special_tokens, text))
        encoding: BatchEncoding = self.tokenizer.batch_encode_plus(
            text, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)
            nopad_mask = ids != self.tokenizer.pad_token_id
            logits: torch.Tensor = self.model(ids, attention_mask=attention_mask)[0]

        for sent_index in range(len(text)):
            sent_nopad_mask = nopad_mask[sent_index]
            # len(tokens) = len(text[sent_index]) + 1
            sent_tokens = [
                tok
                for i, tok in enumerate(encoding.tokens(sent_index))
                if sent_nopad_mask[i] and i != 0
            ]

            # sent_ids.shape = [len(text[sent_index]) + 1]
            sent_ids = ids[sent_index, sent_nopad_mask][1:]
            # logits.shape = [len(text[sent_index]) + 1, vocab_size]
            sent_logits = logits[sent_index, sent_nopad_mask][:-1, :]
            sent_logits[:, self.tokenizer.pad_token_id] = float("-inf")
            # ids_scores.shape = [seq_len + 1]
            sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
            # log_prob.shape = [seq_len + 1]
            sent_log_probs = sent_ids_scores - sent_logits.logsumexp(1)

            sent_log_probs = cast(torch.DoubleTensor, sent_log_probs)
            sent_ids = cast(torch.LongTensor, sent_ids)

            output = (sent_log_probs, sent_ids, sent_tokens)
            outputs.append(output)

        return outputs

class GPT2LM(object):
    """A Language Model (LM) scorer using GPT2.

    This class utilizes the PaddedGPT2LMScorer for scoring sentences.

    Args:
        model_name_or_path (str): Name or path of the model to be used.
        device (str): The device to run the model on. Default is 'cuda'.
        gpu_batch_size (int): The batch size for GPU processing. Default is 20.
    """
    def __init__(self, model_name_or_path, device='cuda', gpu_batch_size=20):
        self.scorer = PaddedGPT2LMScorer(model_name_or_path, device=device, batch_size=gpu_batch_size)

    def get_probs(self, list_of_candidates):
        """Calculates the probabilities for a list of candidate sentences.

        Args:
            list_of_candidates (List[str]): List of candidate sentences to calculate probabilities for.

        Returns:
            List[float]: List of probabilities corresponding to the input sentences.
        """
        scores =  self.scorer.sentence_score(list_of_candidates, log=True)
        scores = [ 1-x for x in scores ]
        return scores
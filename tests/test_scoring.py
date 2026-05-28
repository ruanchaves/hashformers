from types import SimpleNamespace

import pytest
import torch

from hashformers.beamsearch.minicons_lm import MiniconsLM
from hashformers.scoring import transformer_scorers as scorer_module


class FakeBatchEncoding(dict):
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class FakeCausalTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    bos_token = "<s>"
    bos_token_id = 1
    eos_token = "</s>"
    eos_token_id = 2
    padding_side = "right"

    vocab = {
        "a": 3,
        "b": 4,
    }

    def __len__(self):
        return 8

    def add_special_tokens(self, tokens):
        self.pad_token = tokens["pad_token"]

    def __call__(self, texts, return_tensors="pt", padding=True):
        if isinstance(texts, str):
            texts = [texts]

        rows = []
        for text in texts:
            ids = []
            if text.startswith(self.bos_token):
                ids.append(self.bos_token_id)
                text = text[len(self.bos_token) :]
            if text.endswith(self.eos_token):
                text = text[: -len(self.eos_token)]
                has_eos = True
            else:
                has_eos = False

            ids.extend(self.vocab[char] for char in text if char.strip())
            if has_eos:
                ids.append(self.eos_token_id)
            rows.append(ids)

        max_len = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad_len = max_len - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)

        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class FakeMaskedTokenizer:
    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    bos_token_id = 1
    eos_token_id = 2
    mask_token_id = 3
    vocab = {
        "a": 4,
        "b": 5,
    }

    def __call__(self, texts, return_tensors="pt", padding=True):
        if isinstance(texts, str):
            texts = [texts]

        rows = []
        for text in texts:
            ids = [self.cls_token_id]
            ids.extend(self.vocab[char] for char in text if char.strip())
            ids.append(self.sep_token_id)
            rows.append(ids)

        max_len = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad_len = max_len - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)

        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class FakeSeq2SeqTokenizer:
    pad_token_id = 0
    blank_token_id = 1
    vocab = {
        "a": 2,
        "b": 3,
    }

    def __init__(self):
        self.calls = []

    def __call__(self, texts, return_tensors="pt", padding=True):
        if isinstance(texts, str):
            texts = [texts]

        self.calls.append(list(texts))

        rows = []
        for text in texts:
            ids = [self.blank_token_id] if text == "" else [self.vocab[char] for char in text if char.strip()]
            rows.append(ids)

        max_len = max(len(row) for row in rows)
        input_ids = []
        attention_mask = []
        for row in rows:
            pad_len = max_len - len(row)
            input_ids.append(row + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(row) + [0] * pad_len)

        return FakeBatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class FakeCausalModel:
    def __call__(self, input_ids, attention_mask=None):
        vocab_size = 8
        batch_size, sequence_length = input_ids.shape
        logits = torch.full((batch_size, sequence_length, vocab_size), -100.0)

        for row in range(batch_size):
            valid_len = int(attention_mask[row].sum().item())
            valid_ids = input_ids[row, :valid_len]
            for position in range(max(valid_len - 1, 0)):
                logits[row, position, valid_ids[position + 1]] = 0.0

        return SimpleNamespace(logits=logits)


class FakeMaskedModel:
    position_targets = {
        1: 4,
        2: 5,
    }

    def __call__(self, input_ids, attention_mask=None):
        vocab_size = 8
        batch_size, sequence_length = input_ids.shape
        logits = torch.full((batch_size, sequence_length, vocab_size), -100.0)

        for row in range(batch_size):
            masked_position = int((input_ids[row] == 3).nonzero(as_tuple=False)[0].item())
            logits[row, masked_position, self.position_targets[masked_position]] = 0.0

        return SimpleNamespace(logits=logits)


class FakeSeq2SeqModel:
    def __call__(self, input_ids, attention_mask=None, labels=None):
        vocab_size = 8
        batch_size, sequence_length = labels.shape
        logits = torch.full((batch_size, sequence_length, vocab_size), -100.0)

        for row in range(batch_size):
            for position in range(sequence_length):
                label = int(labels[row, position].item())
                if label != -100:
                    logits[row, position, label] = 0.0

        return SimpleNamespace(logits=logits)


def build_incremental_scorer():
    scorer = object.__new__(scorer_module.IncrementalScorer)
    scorer.device = "cpu"
    scorer.gpu_batch_size = 8
    scorer.tokenizer = FakeCausalTokenizer()
    scorer.model = FakeCausalModel()
    return scorer


def build_masked_scorer():
    scorer = object.__new__(scorer_module.MaskedScorer)
    scorer.device = "cpu"
    scorer.gpu_batch_size = 8
    scorer.tokenizer = FakeMaskedTokenizer()
    scorer.model = FakeMaskedModel()
    scorer.excluded_token_ids = {
        scorer.tokenizer.pad_token_id,
        scorer.tokenizer.cls_token_id,
        scorer.tokenizer.sep_token_id,
        scorer.tokenizer.bos_token_id,
        scorer.tokenizer.eos_token_id,
    }
    return scorer


def build_seq2seq_scorer():
    scorer = object.__new__(scorer_module.Seq2SeqScorer)
    scorer.device = "cpu"
    scorer.gpu_batch_size = 8
    scorer.tokenizer = FakeSeq2SeqTokenizer()
    scorer.model = FakeSeq2SeqModel()
    return scorer


def test_incremental_scorer_scores_perfect_sequence():
    scorer = build_incremental_scorer()

    scores = scorer.sequence_score(
        ["ab"],
        reduction=lambda tensor: tensor.sum(0).item(),
        bos_token=True,
        eos_token=True,
    )

    assert scores == [pytest.approx(0.0, abs=1e-6)]


def test_masked_scorer_works_without_batch_encode_plus():
    scorer = build_masked_scorer()

    assert not hasattr(scorer.tokenizer, "batch_encode_plus")

    scores = scorer.sequence_score(
        ["ab"],
        reduction=lambda tensor: tensor.sum(0).item(),
    )

    assert scores == [pytest.approx(0.0, abs=1e-6)]


def test_seq2seq_scorer_uses_blank_source_by_default():
    scorer = build_seq2seq_scorer()

    scores = scorer.sequence_score(
        ["ab"],
        reduction=lambda tensor: tensor.sum(0).item(),
        source_format="blank",
    )

    assert scores == [pytest.approx(0.0, abs=1e-6)]
    assert scorer.tokenizer.calls[0] == [""]
    assert scorer.tokenizer.calls[1] == ["ab"]


def test_masked_scorer_rejects_sequence_classification_checkpoint(monkeypatch):
    monkeypatch.setattr(
        scorer_module.AutoConfig,
        "from_pretrained",
        lambda _: SimpleNamespace(architectures=["XLMRobertaForSequenceClassification"]),
    )
    monkeypatch.setattr(
        scorer_module,
        "_supported_architectures",
        lambda *_args, **_kwargs: ["XLMRobertaForMaskedLM"],
    )

    with pytest.raises(ValueError, match="SequenceClassification"):
        scorer_module.MaskedScorer("demo-model", device="cpu", gpu_batch_size=1)


def test_minicons_lm_accepts_legacy_alias(monkeypatch):
    class FakeIncrementalScorer:
        def __init__(self, model_name_or_path, device, gpu_batch_size):
            self.model = object()
            self.tokenizer = object()

        def sequence_score(self, batch, reduction, **kwargs):
            return [reduction(torch.tensor([0.0])) for _ in batch]

    monkeypatch.setitem(MiniconsLM.scorer_map, "incremental", FakeIncrementalScorer)

    with pytest.deprecated_call():
        lm = MiniconsLM(
            model_name_or_path="demo-model",
            device="cpu",
            gpu_batch_size=2,
            model_type="gpt2",
        )

    assert lm.model_type == "incremental"

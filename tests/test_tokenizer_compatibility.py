from types import SimpleNamespace

from hashformers.beamsearch import minicons_lm


class TransformersFiveTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "encoded"


class TransformersFourTokenizer(TransformersFiveTokenizer):
    def batch_encode_plus(self, *args, **kwargs):
        return "legacy"


def test_adds_batch_encode_plus_when_missing():
    tokenizer = TransformersFiveTokenizer()

    minicons_lm.ensure_tokenizer_compatibility(tokenizer)
    result = tokenizer.batch_encode_plus(["text"], padding="longest")

    assert result == "encoded"
    assert tokenizer.calls == [((["text"],), {"padding": "longest"})]


def test_preserves_existing_batch_encode_plus():
    tokenizer = TransformersFourTokenizer()
    original = tokenizer.batch_encode_plus

    minicons_lm.ensure_tokenizer_compatibility(tokenizer)

    assert tokenizer.batch_encode_plus == original
    assert tokenizer.batch_encode_plus(["text"]) == "legacy"


def test_scorer_receives_compatibility_shim(monkeypatch):
    tokenizer = TransformersFiveTokenizer()

    class FakeScorer:
        def __init__(self, model_name_or_path, device):
            self.tokenizer = tokenizer

    monkeypatch.setattr(
        minicons_lm,
        "scorer",
        SimpleNamespace(MaskedLMScorer=FakeScorer),
    )

    minicons_lm.MiniconsLM("model", "cpu", model_type="MaskedLMScorer")

    assert callable(tokenizer.batch_encode_plus)

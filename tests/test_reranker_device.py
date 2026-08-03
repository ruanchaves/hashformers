from hashformers.beamsearch import model_lm
from hashformers.beamsearch import bert_lm


class RecordingBertLM:
    calls = []

    def __init__(
        self,
        model_name_or_path,
        gpu_batch_size,
        gpu_id,
        device,
        max_gpu_batch_size,
    ):
        self.calls.append({
            "model_name_or_path": model_name_or_path,
            "gpu_batch_size": gpu_batch_size,
            "gpu_id": gpu_id,
            "device": device,
            "max_gpu_batch_size": max_gpu_batch_size,
        })


def test_bert_model_receives_requested_device(monkeypatch):
    monkeypatch.setattr(model_lm, "BertLM", RecordingBertLM)

    model_lm.ModelLM(
        model_name_or_path="bert-model",
        model_type="bert",
        device="cpu",
        gpu_batch_size=8,
        gpu_id=2,
        max_gpu_batch_size=128,
    )

    assert RecordingBertLM.calls[-1] == {
        "model_name_or_path": "bert-model",
        "gpu_batch_size": 8,
        "gpu_id": 2,
        "device": "cpu",
        "max_gpu_batch_size": 128,
    }


def test_explicit_cuda_device_is_preserved(monkeypatch):
    monkeypatch.setattr(model_lm, "BertLM", RecordingBertLM)

    model_lm.ModelLM(
        model_name_or_path="bert-model",
        model_type="bert",
        device="cuda:1",
        gpu_batch_size=8,
    )

    assert RecordingBertLM.calls[-1]["device"] == "cuda:1"


def test_bert_lm_passes_device_to_scorer_wrapper(monkeypatch):
    received = {}

    def record_init(self, **kwargs):
        received.update(kwargs)

    monkeypatch.setattr(bert_lm.MiniconsLM, "__init__", record_init)

    bert_lm.BertLM("bert-model", device="cpu")

    assert received["device"] == "cpu"

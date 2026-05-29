import math
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional, Sequence

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

SUPPORTED_MODEL_TYPES = ("incremental", "masked", "seq2seq")
MODEL_TYPE_ALIASES = {
    "gpt2": "incremental",
    "bert": "masked",
    "IncrementalLMScorer": "incremental",
    "MaskedLMScorer": "masked",
    "Seq2SeqScorer": "seq2seq",
}


def canonicalize_model_type(model_type: Optional[str]) -> Optional[str]:
    if model_type is None:
        return None

    if model_type in MODEL_TYPE_ALIASES:
        canonical = MODEL_TYPE_ALIASES[model_type]
        warnings.warn(
            f"`{model_type}` is deprecated; use `{canonical}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return canonical

    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(list(SUPPORTED_MODEL_TYPES) + list(MODEL_TYPE_ALIASES))
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported values: {supported}.")

    return model_type


def _coerce_batch(batch: Sequence[str]) -> List[str]:
    if isinstance(batch, str):
        return [batch]
    return list(batch)


def _iter_batches(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    effective_batch_size = max(int(batch_size or 1), 1)
    for start in range(0, len(items), effective_batch_size):
        yield list(items[start : start + effective_batch_size])


def _load_tokenizer(model_name_or_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name_or_path)


def _load_model(auto_model_class, model_name_or_path: str, device: str):
    kwargs = {"return_dict": True}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = auto_model_class.from_pretrained(model_name_or_path, **kwargs)
    if device != "auto":
        model.to(device)
    model.eval()
    return model


def _supported_architectures(auto_model_class, config) -> List[str]:
    mapping = getattr(auto_model_class, "_model_mapping", None)
    if mapping is None:
        return []

    for config_cls in mapping.keys():
        if isinstance(config, config_cls):
            model_entry = mapping[config_cls]
            if not isinstance(model_entry, (list, tuple)):
                model_entry = (model_entry,)

            names = []
            for item in model_entry:
                names.append(getattr(item, "__name__", str(item)))
            return names

    return []


def _validate_model_compatibility(
    model_name_or_path: str,
    requested_model_type: str,
    auto_model_class,
    expected_family: str,
):
    config = AutoConfig.from_pretrained(model_name_or_path)
    supported_architectures = _supported_architectures(auto_model_class, config)
    configured_architectures = list(getattr(config, "architectures", None) or [])

    if supported_architectures and configured_architectures:
        if set(configured_architectures).isdisjoint(supported_architectures):
            supported = ", ".join(supported_architectures)
            configured = ", ".join(configured_architectures)
            raise ValueError(
                f"Model '{model_name_or_path}' is not compatible with model_type='{requested_model_type}'. "
                f"Detected architecture(s): {configured}. Expected a checkpoint loadable with {expected_family} "
                f"(supported architectures for this config: {supported})."
            )

    elif not supported_architectures:
        config_name = type(config).__name__
        raise ValueError(
            f"Model '{model_name_or_path}' is not compatible with model_type='{requested_model_type}'. "
            f"Detected config: {config_name}. Expected a checkpoint loadable with {expected_family}."
        )

    return config


def _gather_scores(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = logits.log_softmax(-1)
    row_indices = torch.arange(target_ids.shape[0], device=logits.device)
    return log_probs[row_indices, target_ids]


def _finalize_token_scores(scores: torch.Tensor, prob: bool, base_two: bool) -> torch.Tensor:
    if prob and base_two:
        raise ValueError("Cannot request base-two scores and probabilities at the same time.")

    if base_two:
        return scores / math.log(2.0)
    if prob:
        return scores.exp()
    return scores


class BaseTransformerScorer(ABC):
    auto_model_class = None
    expected_family = ""

    def __init__(self, model_name_or_path: str, device: str = "cpu", gpu_batch_size: int = 20):
        self.model_name_or_path = model_name_or_path
        self.device = device or "cpu"
        self.gpu_batch_size = max(int(gpu_batch_size or 1), 1)
        self.config = _validate_model_compatibility(
            model_name_or_path=model_name_or_path,
            requested_model_type=self.model_type,
            auto_model_class=self.auto_model_class,
            expected_family=self.expected_family,
        )
        self.tokenizer = _load_tokenizer(model_name_or_path)
        self.model = _load_model(self.auto_model_class, model_name_or_path, self.device)

    @property
    @abstractmethod
    def model_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sequence_score(self, batch, reduction: Callable = lambda x: x.mean(0).item(), **kwargs):
        raise NotImplementedError


class IncrementalScorer(BaseTransformerScorer):
    auto_model_class = AutoModelForCausalLM
    expected_family = "AutoModelForCausalLM"

    @property
    def model_type(self) -> str:
        return "incremental"

    def __init__(self, model_name_or_path: str, device: str = "cpu", gpu_batch_size: int = 20):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.padding_side == "left":
            self.tokenizer.padding_side = "right"

    def _encode(self, batch, bos_token: bool = False, eos_token: bool = False):
        sentences = _coerce_batch(batch)

        if bos_token and self.tokenizer.bos_token is not None:
            sentences = [self.tokenizer.bos_token + sentence for sentence in sentences]
        if eos_token and self.tokenizer.eos_token is not None:
            sentences = [sentence + self.tokenizer.eos_token for sentence in sentences]

        encoded = self.tokenizer(sentences, return_tensors="pt", padding=True)
        if "token_type_ids" in encoded:
            encoded.pop("token_type_ids")
        return encoded

    def sequence_score(
        self,
        batch,
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        bos_token: bool = False,
        eos_token: bool = False,
    ):
        encoded = self._encode(batch, bos_token=bos_token, eos_token=eos_token)
        model_inputs = encoded if self.device == "auto" else encoded.to(self.device)

        with torch.no_grad():
            logits = self.model(**model_inputs).logits

        sequence_scores = []
        for index in range(model_inputs["input_ids"].shape[0]):
            input_ids = model_inputs["input_ids"][index].to(logits.device)
            attention_mask = model_inputs["attention_mask"][index].to(logits.device).bool()
            valid_ids = input_ids[attention_mask]

            if valid_ids.shape[0] <= 1:
                token_scores = torch.zeros(1, device=logits.device)
            else:
                target_ids = valid_ids[1:]
                step_logits = logits[index][attention_mask][:-1]
                token_scores = _gather_scores(step_logits, target_ids)

            token_scores = _finalize_token_scores(token_scores, prob=prob, base_two=base_two)
            sequence_scores.append(reduction(token_scores))

        return sequence_scores


class MaskedScorer(BaseTransformerScorer):
    auto_model_class = AutoModelForMaskedLM
    expected_family = "AutoModelForMaskedLM"

    @property
    def model_type(self) -> str:
        return "masked"

    def __init__(self, model_name_or_path: str, device: str = "cpu", gpu_batch_size: int = 20):
        super().__init__(
            model_name_or_path=model_name_or_path,
            device=device,
            gpu_batch_size=gpu_batch_size,
        )
        if self.tokenizer.mask_token_id is None:
            raise ValueError(
                f"Model '{model_name_or_path}' does not expose a mask token and cannot be used as a masked reranker."
            )

        self.excluded_token_ids = {
            token_id
            for token_id in (
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id,
            )
            if token_id is not None
        }

    def _token_scores_for_sentence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid_positions = [
            position
            for position, token_id in enumerate(input_ids.tolist())
            if attention_mask[position].item() and token_id not in self.excluded_token_ids
        ]

        if not valid_positions:
            return torch.zeros(1)

        masked_input_ids = input_ids.unsqueeze(0).repeat(len(valid_positions), 1)
        position_tensor = torch.tensor(valid_positions, dtype=torch.long)
        masked_input_ids[torch.arange(len(valid_positions)), position_tensor] = self.tokenizer.mask_token_id
        repeated_attention = attention_mask.unsqueeze(0).repeat(len(valid_positions), 1)
        target_ids = input_ids[position_tensor]

        chunks = []
        for start in range(0, len(valid_positions), self.gpu_batch_size):
            stop = start + self.gpu_batch_size
            batch_input_ids = masked_input_ids[start:stop]
            batch_attention = repeated_attention[start:stop]
            batch_positions = position_tensor[start:stop]
            batch_targets = target_ids[start:stop]

            if self.device != "auto":
                batch_input_ids = batch_input_ids.to(self.device)
                batch_attention = batch_attention.to(self.device)
                batch_positions = batch_positions.to(self.device)
                batch_targets = batch_targets.to(self.device)

            with torch.no_grad():
                logits = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention,
                ).logits

            selected_logits = logits[
                torch.arange(batch_targets.shape[0], device=logits.device),
                batch_positions,
            ]
            chunks.append(_gather_scores(selected_logits, batch_targets))

        return torch.cat(chunks)

    def sequence_score(
        self,
        batch,
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
    ):
        encoded = self.tokenizer(_coerce_batch(batch), return_tensors="pt", padding=True)

        sequence_scores = []
        for input_ids, attention_mask in zip(encoded["input_ids"], encoded["attention_mask"]):
            token_scores = self._token_scores_for_sentence(input_ids, attention_mask)
            token_scores = _finalize_token_scores(token_scores, prob=prob, base_two=base_two)
            sequence_scores.append(reduction(token_scores))

        return sequence_scores


class Seq2SeqScorer(BaseTransformerScorer):
    auto_model_class = AutoModelForSeq2SeqLM
    expected_family = "AutoModelForSeq2SeqLM"

    @property
    def model_type(self) -> str:
        return "seq2seq"

    def _resolve_sources(self, batch, source_format: str = "blank", source=None) -> List[str]:
        targets = _coerce_batch(batch)

        if source is not None:
            sources = _coerce_batch(source)
            if len(sources) != len(targets):
                raise ValueError("Custom seq2seq sources must match the batch length.")
            return sources

        if source_format == "blank":
            return [""] * len(targets)
        if source_format == "copy":
            return list(targets)

        raise ValueError(f"Unsupported source_format '{source_format}'.")

    def sequence_score(
        self,
        batch,
        reduction: Callable = lambda x: x.mean(0).item(),
        prob: bool = False,
        base_two: bool = False,
        source_format: str = "blank",
        source=None,
    ):
        targets = _coerce_batch(batch)
        sources = self._resolve_sources(targets, source_format=source_format, source=source)

        sequence_scores = []
        for batch_indices in _iter_batches(list(range(len(targets))), self.gpu_batch_size):
            batch_sources = [sources[index] for index in batch_indices]
            batch_targets = [targets[index] for index in batch_indices]

            source_encoded = self.tokenizer(batch_sources, return_tensors="pt", padding=True)
            target_encoded = self.tokenizer(batch_targets, return_tensors="pt", padding=True)

            labels = target_encoded["input_ids"].clone()
            labels[target_encoded["attention_mask"] == 0] = -100

            if self.device != "auto":
                source_encoded = source_encoded.to(self.device)
                labels = labels.to(self.device)
                target_input_ids = target_encoded["input_ids"].to(self.device)
            else:
                target_input_ids = target_encoded["input_ids"]

            with torch.no_grad():
                logits = self.model(**source_encoded, labels=labels).logits

            for row_index in range(len(batch_indices)):
                valid_mask = labels[row_index] != -100
                if not torch.any(valid_mask):
                    token_scores = torch.zeros(1, device=logits.device)
                else:
                    target_ids = target_input_ids[row_index][valid_mask]
                    step_logits = logits[row_index][valid_mask]
                    token_scores = _gather_scores(step_logits, target_ids)

                token_scores = _finalize_token_scores(token_scores, prob=prob, base_two=base_two)
                sequence_scores.append(reduction(token_scores))

        return sequence_scores

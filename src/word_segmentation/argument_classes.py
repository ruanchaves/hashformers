from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class TranslationArguments:

    translation_model_name_or_path: str = field(
        default="Helsinki-NLP/opus-mt-en-es"
    )

    translation_model_batch_size: int = field(
        default=1000
    )

    translation_model_device: str = field(
        default="cuda"
    )

@dataclass
class TranslationDataArguments:

    log_level: str = field(
        default='INFO'
    )

    dataset_reader: str = field(
        default='./umsab.py'
    )

    dataset_load_path: Optional[str] = field(
        default=None
    )

    dataset_save_path: Optional[str] = field(
        default=None
    )

    dataset_base_url: str = field(
        default=None
    )

    content_field: str = field(
        default = "content"
    )

    translation_field: str = field(
        default = "translation"
    )

    content_replaced_hashtags_field: str = field(
        default="content_replaced_hashtags"
    )

    translation_replaced_hashtags_field: str = field(
        default="translation_replaced_hashtags"
    )

    translation_segmented_hashtags_field: str = field(
        default="translation_segmented_hashtags"
    )

    sample: Optional[int] = field(
        default=None
    ) 

@dataclass
class DataArguments:

    do_eval: bool = field(
        default=True
    )

    log_level: str = field(
        default='INFO'
    )

    dataset_reader: str = field(
        default='./semeval2017.py'
    )

    dataset_load_path: Optional[str] = field(
        default=None
    )

    dataset_save_path: Optional[str] = field(
        default=None
    )

    dataset_url: str = field(
        default=None
    )

    split: str = field(
        default='test'
    )

    content_field: str = field(
        default = "content"
    )

    predictions_field: Optional[str] = field(
        default="predictions"
    )

    segmented_content_field: Optional[str] = field(
        default="segmented_content"
    )

    segmented_predictions_field: str = field(
        default="segmented_predictions"
    )

    label_field: str = field(
        default = "polarity"
    )

    sample: Optional[int] = field(
        default=None
    )

@dataclass
class TextClassificationArguments:

    run_classifier: bool = field(
        default=True
    )

    sentiment_model: str = field(
        default="finiteautomata/beto-sentiment-analysis"
    )

    sentiment_model_device: str = field(
        default="cuda"
    )

    batch_size: int = field(
        default=1
    )

    metrics: str = field(
        default="./sentiment_metrics.py"
    )

    prune_layers: Optional[List[int]] = field(
        default=None
    )

    max_length: int = field(
        default=512
    )

@dataclass
class WordSegmenterArguments:

    hashtag_dict_load_path: Optional[str] = field(
        default=None
    )

    hashtag_dict_save_path: Optional[str] = field(
        default=None
    )

    run_segmenter: bool = field(
        default=True
    )

    decoder_model_name_or_path: str = field(
        default="DeepESP/gpt2-spanish"
    )

    decoder_model_type: str = field(
        default="gpt2"
    )

    decoder_device: str = field(
        default="cuda"
    )

    decoder_gpu_batch_size: int = field(
        default=1
    )

    encoder_model_name_or_path: str = field(
        default="dccuchile/bert-base-spanish-wwm-uncased"
    )

    encoder_model_type: str = field(
        default="bert"
    )

    spacy_model: str = field(
        default="es_core_news_sm"
    )

    topk: int = field(
        default=20
    )

    steps: int = field(
        default=13
    )

    alpha: float = field(
        default=0.222
    )

    beta: float = field(
        default=0.111
    )

    use_encoder = field(
        default=True
    )
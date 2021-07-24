import json
from dataclasses import dataclass, field
import logging
import os
import sys
from torch import nn
import torch
from contextlib import suppress

import datasets
from datasets import (
    load_dataset,
    load_metric
)

import transformers
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser
)

from word_segmentation import WordSegmenter
from typing import Optional 
import copy
import functools

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:

    do_eval: bool = field(
        default=True
    )

    log_level: str = field(
        default='INFO'
    )

    dataset_reader: str = field(
        default='./tass_reader.py'
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

    label_field: str = field(
        default = "polarity"
    )

    sample: Optional[int] = field(
        default=None
    )

    hashtag_only: bool = field(
        default=True
    )

    skip_neutral: bool = field(
        default=True
    )

    neutral_field: str = field(
        default="2"
    )

    positive_field: str = field(
        default="1"
    )

    negative_field: str = field(
        default="0"
    )

@dataclass
class TextClassificationArguments:

    run_classifier: bool = field(
        default=True
    )

    sentiment_model: str = field(
        default="finiteautomata/beto-sentiment-analysis"
    )

    sentiment_model_device: int = field(
        default=0
    )

    batch_size: int = field(
        default=1
    )

    metrics: str = field(
        default="./sentiment_metrics.py"
    )

    prune_layers: Optional[int] = field(
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

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def process_rows(
        batch, 
        model=None, 
        tokenizer=None, 
        content_field="content", 
        predictions_field="predictions",
        max_length=512):
    sentences = batch[content_field]
    tokens = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=max_length
        )
    logits = model(**tokens).logits
    softmax_logits = torch.softmax(logits, dim=1)
    _, preds = torch.max(softmax_logits, 1)
    preds = preds.tolist()
    preds = [ str(x) for x in preds ]
    batch.update({predictions_field: preds})
    return batch

def eval_dataset(
    data,
    split="test",
    reference_field="polarity",
    predictions_field="predictions",
    metric="./sentiment_metrics.py"):
    predictions = data[split][predictions_field]
    references = data[split][reference_field]
    metric = load_metric(metric)
    eval_results = metric.compute(
        predictions=predictions,
        references=references
    )
    return eval_results


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    model_type = rgetattr(model, "config.model_type")
    
    oldModuleList = None
    keys = ['encoder', 'transformer']

    for item in keys:
        with suppress(AttributeError):
            oldModuleList = rgetattr(model, f"{model_type}.{item}.layer")

    if not oldModuleList:
        raise NotImplementedError

    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)

    for item in keys:
        with suppress(AttributeError):
            rsetattr(copyOfModel, f"{model_type}.{item}.layer", newModuleList)

    return copyOfModel

def main():
    parser = HfArgumentParser((TextClassificationArguments, WordSegmenterArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
            class_args, ws_args, data_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        class_args, ws_args, data_args = \
            parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(data_args.log_level)
    datasets.utils.logging.set_verbosity(data_args.log_level)
    transformers.utils.logging.set_verbosity(data_args.log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if ws_args.hashtag_dict_load_path:
        with open(ws_args.hashtag_dict_load_path, 'r') as f:
            main_hashtag_dict = json.load(f)
    else:
        main_hashtag_dict = {}

    if ws_args.run_segmenter:
        ws = WordSegmenter(
            decoder_model_name_or_path=ws_args.decoder_model_name_or_path,
            decoder_model_type=ws_args.decoder_model_type,
            decoder_device=ws_args.decoder_device,
            decoder_gpu_batch_size=ws_args.decoder_gpu_batch_size,
            encoder_model_name_or_path=ws_args.encoder_model_name_or_path,
            encoder_model_type=ws_args.encoder_model_type,
            spacy_model=ws_args.spacy_model
        )
    else:
        ws = None

    if data_args.dataset_load_path:
        data = datasets.load_from_disk(data_args.dataset_load_path)
    else:
        data = load_dataset(data_args.dataset_reader, url=data_args.dataset_url)

    if data_args.hashtag_only:
        data = data.filter(lambda x: x["has_hashtag"])

    if data_args.skip_neutral:
        data = data.filter(lambda x: x["polarity"] != data_args.neutral_field)

    if data_args.sample:
        data[data_args.split] = data[data_args.split]\
            .select([i for i in range(0, data_args.sample)])

    label_mapping = {
        "0": data_args.negative_field,
        "1": data_args.positive_field,
        "2": data_args.neutral_field
    }
    def map_labels(row, mapping = None):
        row["polarity"] = mapping[row["polarity"]]
        return row
    data = data.map(
        map_labels,
        fn_kwargs={
            "mapping": label_mapping
        })

    split_length = data[data_args.split].shape[0]

    def segment_content(
        batch, 
        segmenter=None, 
        content_field="content", 
        segmented_content_field="segmented_content",
        dictionary=main_hashtag_dict):

        sentences = batch[content_field]
        segmented_content, hashtag_dict = segmenter.process_hashtags(
            sentences,
            dictionary=dictionary)
        segmented_content = [ " ".join(x) for x in segmented_content]
        batch.update({segmented_content_field: segmented_content})
        main_hashtag_dict.update(hashtag_dict)
        return batch

    if ws_args.run_segmenter:
        data = data.map(
            segment_content,
            fn_kwargs={
                "segmenter": ws,
                "content_field": data_args.content_field
            },
            batched=True, 
            batch_size=split_length
        )

    if class_args.run_classifier:
        original_model = AutoModelForSequenceClassification.from_pretrained(class_args.sentiment_model)
        if class_args.prune_layers:
            model_range = class_args.prune_layers
        else:
            model_range = 1
        tokenizer = AutoTokenizer.from_pretrained(class_args.sentiment_model)

    for idx in range(0, model_range):

        if class_args.prune_layers:
            model = deleteEncodingLayers(original_model, idx)
        else:
            model = original_model

        if class_args.run_classifier:
            data = data.map(
                process_rows, 
                fn_kwargs={
                    "model": model,
                    "tokenizer": tokenizer,
                    "content_field": data_args.content_field
                },
                batched=True, 
                batch_size=class_args.batch_size)

        if class_args.run_classifier:
            data = data.map(
                process_rows, 
                fn_kwargs={
                    "model": model,
                    "tokenizer": tokenizer,
                    "content_field": "segmented_content",
                    "predictions_field": "segmented_predictions"
                },
                batched=True, 
                batch_size=class_args.batch_size)

        if data_args.do_eval:

            if data_args.hashtag_only:
                subset_evaluation = eval_dataset(
                    data,
                    metric=class_args.metrics
                )
                full_evaluation = None
            else:
                data_subset = data.filter(lambda x: x['has_hashtag'])
                subset_evaluation = eval_dataset(data_subset)
                full_evaluation = eval_dataset(data)

            if data_args.hashtag_only:
                subset_evaluation_after_segmentation = eval_dataset(
                    data,
                    predictions_field="segmented_predictions"
                )
                full_evaluation_after_segmentation = None
            else:
                data_subset = data.filter(lambda x: x['has_hashtag'])
                subset_evaluation_after_segmentation = eval_dataset(
                    data_subset,
                    predictions_field="segmented_predictions"
                )
                full_evaluation_after_segmentation = eval_dataset(
                    data,
                    predictions_field="segmented_predictions"
                )

            log_args = {}
            for item in [class_args, ws_args, data_args]:
                log_args.update(vars(item))

            logger.info("Parameters:")
            logger.info("%s", log_args)

            logger.info("idx:")
            logger.info("%s", str(idx))

            logger.info("Full dataset evaluation:")
            logger.info("%s", full_evaluation)

            logger.info("Hashtag subset evaluation:")
            logger.info("%s", subset_evaluation)

            logger.info("Hashtag subset evaluation after hashtag segmentation:")
            logger.info("%s", subset_evaluation_after_segmentation)

            logger.info("Full dataset evaluation after hashtag segmentation:")
            logger.info("%s", full_evaluation_after_segmentation)

    if data_args.dataset_save_path:
        data.save_to_disk(data_args.dataset_save_path)

    if ws_args.hashtag_dict_save_path:
        with open(ws_args.hashtag_dict_save_path, 'w') as f:
            json.dump(main_hashtag_dict, f)

if __name__ == '__main__':
    main()
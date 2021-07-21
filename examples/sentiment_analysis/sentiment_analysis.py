from dataclasses import dataclass, field
import logging
import os
import sys
from torch import nn

import datasets
from datasets import (
    load_dataset,
    load_metric
)

import transformers
from transformers import (
    AutoConfig,
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

    log_level: str = field(
        default='INFO'
    )

    dataset_reader: str = field(
        default='./tass_reader.py'
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

    output_dir: str = field(
        default="./output"
    )

    load_cache_from_output_dir: bool = field(
        default=True
    )

    sample: Optional[int] = field(
        default=None
    )

    hashtag_only: bool = field(
        default=True
    )

@dataclass
class TextClassificationArguments:

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

@dataclass
class WordSegmenterArguments:

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

    ws = WordSegmenter(
        decoder_model_name_or_path=ws_args.decoder_model_name_or_path,
        decoder_model_type=ws_args.decoder_model_type,
        decoder_device=ws_args.decoder_device,
        decoder_gpu_batch_size=ws_args.decoder_gpu_batch_size,
        encoder_model_name_or_path=ws_args.encoder_model_name_or_path,
        encoder_model_type=ws_args.encoder_model_type,
        spacy_model=ws_args.spacy_model
    )

    def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
        model_type = rgetattr(model, "config.model_type")
        oldModuleList = rgetattr(model, f"{model_type}.encoder.layer")
        newModuleList = nn.ModuleList()

        # Now iterate over all layers, only keepign only the relevant layers.
        for i in range(0, num_layers_to_keep):
            newModuleList.append(oldModuleList[i])

        # create a copy of the model, modify it with the new list, and return
        copyOfModel = copy.deepcopy(model)
        rsetattr(copyOfModel, f"{model_type}.encoder.layer", newModuleList)

        return copyOfModel

    model = AutoModelForSequenceClassification.from_pretrained(class_args.sentiment_model)
    if class_args.prune_layers:
        model = deleteEncodingLayers(model, class_args.prune_layers)

    tokenizer = AutoTokenizer.from_pretrained(class_args.sentiment_model)
    classifier = pipeline("sentiment-analysis", 
        model=model,
        tokenizer=tokenizer,
        device=class_args.sentiment_model_device)

    data = load_dataset(data_args.dataset_reader, url=data_args.dataset_url)

    if data_args.hashtag_only:
        data = data.filter(lambda x: x["has_hashtag"])

    if data_args.sample:
        data[data_args.split] = data[data_args.split]\
            .select([i for i in range(0, data_args.sample)])

    split_length = data[data_args.split].shape[0]

    def process_rows(batch, classifier=None, content_field="content", predictions_field="predictions"):
        sentences = batch[content_field]
        labels = classifier(sentences)
        batch.update({predictions_field: labels})
        return batch

    def segment_content(batch, segmenter=None, content_field="content", segmented_content_field="segmented_content"):
        sentences = batch[content_field]
        segmented_content = segmenter.process_hashtags(sentences)
        segmented_content = [ " ".join(x) for x in segmented_content]
        batch.update({segmented_content_field: segmented_content})
        return batch

    def eval_dataset(
        data,
        split="test",
        reference_field="polarity",
        predictions_field="predictions",
        metric="./sentiment_metrics.py"):
        predictions = data[split][predictions_field]
        predictions = [x["label"] for x in predictions]
        references = data[split][reference_field]
        metric = load_metric(metric)
        eval_results = metric.compute(
            predictions=predictions,
            references=references
        )
        return eval_results

    data = data.map(
        process_rows, 
        fn_kwargs={
            "classifier": classifier,
            "content_field": data_args.content_field
        },
        batched=True, 
        batch_size=class_args.batch_size)

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

    data = data.map(
        segment_content,
        fn_kwargs={
            "segmenter": ws,
            "content_field": data_args.content_field
        },
        batched=True, 
        batch_size=split_length
    )

    data = data.map(
        process_rows, 
        fn_kwargs={
            "classifier": classifier,
            "content_field": "segmented_content",
            "predictions_field": "segmented_predictions"
        },
        batched=True, 
        batch_size=class_args.batch_size)

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

    logger.info("Full dataset evaluation:")
    logger.info("%s", full_evaluation)

    logger.info("Hashtag subset evaluation:")
    logger.info("%s", subset_evaluation)

    logger.info("Hashtag subset evaluation after hashtag segmentation:")
    logger.info("%s", subset_evaluation_after_segmentation)

    logger.info("Full dataset evaluation after hashtag segmentation:")
    logger.info("%s", full_evaluation_after_segmentation)


if __name__ == '__main__':
    main()
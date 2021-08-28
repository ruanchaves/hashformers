import json
from dataclasses import dataclass, field
import logging
import os
import sys
from torch import nn
import torch
from contextlib import suppress
from pythonjsonlogger import jsonlogger

import datasets

from datasets import (
    load_dataset,
    load_metric
)

import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser
)

from word_segmentation import (
    WordSegmenter,
    DataArguments,
    TextClassificationArguments,
    WordSegmenterArguments
)

import copy
import functools

logger = logging.getLogger(__name__)

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
        max_length=512,
        device="cuda",
        return_shape=False):
    sentences = batch[content_field]
    tokens = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=max_length
        )
    tokens = tokens.to(device)
    logits = model(**tokens).logits
    logits = logits.to("cpu")
    softmax_logits = torch.softmax(logits, dim=1)

    if return_shape:
        return logits.shape[1]

    _, preds = torch.max(softmax_logits, 1)
    preds = preds.tolist()
    if logits.shape[1] == 2:
        for idx, item in enumerate(preds):
            if item == 1:
                preds[idx] = 2
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

    logHandler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    # Setup logging
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

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
            segmenter_model_name_or_path=ws_args.segmenter_model_name_or_path,
            segmenter_model_type=ws_args.segmenter_model_type,
            segmenter_device=ws_args.segmenter_device,
            segmenter_gpu_batch_size=ws_args.segmenter_gpu_batch_size,
            reranker_model_name_or_path=ws_args.reranker_model_name_or_path,
            reranker_model_type=ws_args.reranker_model_type,
            spacy_model=ws_args.spacy_model
        )
    else:
        ws = None

    if data_args.dataset_load_path:
        logger.info("Loading dataset from disk.")
        data = datasets.load_from_disk(data_args.dataset_load_path)
    else:
        logger.info("Loading dataset with reader.")
        data = load_dataset(data_args.dataset_reader, url=data_args.dataset_url)

    if data_args.sample:
        data[data_args.split] = data[data_args.split]\
            .select([i for i in range(0, data_args.sample)])

    split_length = data[data_args.split].shape[0]

    def segment_content(
        batch, 
        segmenter=None, 
        content_field="content", 
        segmented_content_field="segmented_content",
        topk=20,
        steps=13,
        alpha=0.222,
        beta=0.111,
        use_reranker=True,
        dictionary=None):
        sentences = batch[content_field]
        segmented_content, hashtag_dict = segmenter.process_hashtags(
            sentences,
            topk=topk,
            steps=steps,
            alpha=alpha,
            beta=beta,
            use_reranker=use_reranker,
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
                "content_field": data_args.content_field,
                "segmented_content_field": data_args.segmented_content_field,
                "topk": ws_args.topk,
                "steps": ws_args.steps,
                "alpha": ws_args.alpha,
                "beta": ws_args.beta
            },
            batched=True, 
            batch_size=split_length
        )

    if class_args.run_classifier:
        original_model = AutoModelForSequenceClassification.from_pretrained(class_args.sentiment_model)
        if class_args.prune_layers:
            model_range = len(class_args.prune_layers)
        else:
            model_range = 1
        tokenizer = AutoTokenizer.from_pretrained(class_args.sentiment_model)

        for idx in range(0, model_range):
            if class_args.prune_layers:
                logger.info(f"Processing predictions for layer {class_args.prune_layers[idx]}.")
                model = deleteEncodingLayers(original_model, class_args.prune_layers[idx])
            else:
                model = original_model

            model.to(class_args.sentiment_model_device)

            num_labels = process_rows(
                {"content": ["hello world"]}, 
                model=model,
                tokenizer=tokenizer,
                content_field="content",
                return_shape=True
            )
            if num_labels == 2:
                data = data.filter(lambda x: x[data_args.label_field]!="1")
            elif num_labels == 3:
                pass
            else:
                raise NotImplementedError

            if class_args.run_classifier and data_args.predictions_field:
                logger.info("Writing predictions on the dataset.")
                data = data.map(
                    process_rows, 
                    fn_kwargs={
                        "model": model,
                        "tokenizer": tokenizer,
                        "content_field": data_args.content_field,
                        "predictions_field": data_args.predictions_field
                    },
                    batched=True, 
                    batch_size=class_args.batch_size,
                    keep_in_memory=True)

            if class_args.run_classifier and data_args.segmented_predictions_field:
                logger.info("Writing processed predictions on the dataset.")
                data = data.map(
                    process_rows, 
                    fn_kwargs={
                        "model": model,
                        "tokenizer": tokenizer,
                        "content_field": data_args.segmented_content_field,
                        "predictions_field": data_args.segmented_predictions_field
                    },
                    batched=True, 
                    batch_size=class_args.batch_size,
                    keep_in_memory=True)

            if data_args.do_eval:

                data_subset = data.filter(lambda x: x['has_hashtag'])

                dataset_evaluation_params = {
                    "split": data_args.split,
                    "reference_field": data_args.label_field,
                    "metric": class_args.metrics
                }

                if data_args.predictions_field:
                    
                    subset_evaluation = eval_dataset(
                        data_subset,
                        predictions_field=data_args.predictions_field,
                        **dataset_evaluation_params)
                    subset_evaluation.update({
                        "eval": "subset_evaluation"
                    })

                    full_evaluation = eval_dataset(
                        data,
                        predictions_field=data_args.predictions_field,
                        **dataset_evaluation_params)
                    full_evaluation.update({
                        "eval": "full_evaluation"
                    })

                if data_args.segmented_predictions_field:
                    
                    subset_evaluation_after_segmentation = eval_dataset(
                        data_subset,
                        predictions_field=data_args.segmented_predictions_field,
                        **dataset_evaluation_params)
                    subset_evaluation_after_segmentation.update({
                        "eval": "subset_evaluation_after_segmentation"
                    })

                    full_evaluation_after_segmentation = eval_dataset(
                        data,
                        predictions_field=data_args.segmented_predictions_field,
                        **dataset_evaluation_params)
                    full_evaluation_after_segmentation.update({
                        "eval": "full_evaluation_after_segmentation"
                    })

                log_args = {}
                for item in [class_args, ws_args, data_args]:
                    log_args.update(vars(item))

                for item in [
                    full_evaluation,
                    subset_evaluation,
                    subset_evaluation_after_segmentation,
                    full_evaluation_after_segmentation]:

                    item.update(log_args)
                    if class_args.prune_layers:
                        item.update({"current_layer": class_args.prune_layers[idx]})

                    logger.info(item)

    if data_args.dataset_save_path:
        data.save_to_disk(data_args.dataset_save_path)

    if ws_args.hashtag_dict_save_path:
        with open(ws_args.hashtag_dict_save_path, 'w') as f:
            json.dump(main_hashtag_dict, f)

if __name__ == '__main__':
    main()
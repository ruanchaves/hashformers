from collections import namedtuple
from datasets import load_dataset
import logging
import datasets
import json
import sys
import os
import copy
import shutil
import datetime
import torch
from pythonjsonlogger import jsonlogger
from tqdm import tqdm
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    HfArgumentParser
)

from word_segmentation import (
    Translator,
    TranslationArguments,
    TranslationDataArguments,
    WordSegmenterArguments
)

logger = logging.getLogger(__name__)

def main():

    parser = HfArgumentParser((TranslationArguments, TranslationDataArguments, WordSegmenterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
            translation_args, data_args, ws_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        translation_args, data_args, ws_args = \
            parser.parse_args_into_dataclasses()

    logHandler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter()
    logHandler.setFormatter(formatter)
    logger.addHandler(logHandler)

    translation_kwargs = {
        "batched": True,
        "batch_size": translation_args.translation_model_batch_size
    }

    ws_kwargs = {
        "topk": ws_args.topk,
        "steps": ws_args.steps,
        "alpha": ws_args.alpha,
        "beta": ws_args.beta,
        "use_reranker": ws_args.use_reranker
    }

    if ws_args.hashtag_dict_load_path:
        with open(ws_args.hashtag_dict_load_path, 'r') as f:
            main_hashtag_dict = json.load(f)
    else:
        main_hashtag_dict = {}

    translation_tokenizer = MarianTokenizer.from_pretrained(
        translation_args.translation_model_name_or_path
    )

    translation_model = MarianMTModel.from_pretrained(
        translation_args.translation_model_name_or_path
    )

    translation_model.to(
        translation_args.translation_model_device
    )

    assert translation_tokenizer != None

    model = Translator(
        translation_model=translation_model,
        translation_tokenizer=translation_tokenizer,
        segmenter_model_name_or_path=ws_args.segmenter_model_name_or_path,
        segmenter_model_type=ws_args.segmenter_model_type,
        segmenter_device=ws_args.segmenter_device,
        segmenter_gpu_batch_size=ws_args.segmenter_gpu_batch_size,
        reranker_model_name_or_path=ws_args.reranker_model_name_or_path,
        reranker_model_type=ws_args.reranker_model_type,
        spacy_model=ws_args.spacy_model        
    )

    if data_args.dataset_load_path:
        logger.info("Loading dataset from disk.")
        global_data = datasets.load_from_disk(data_args.dataset_load_path)
    else:
        logger.info("Loading dataset with reader.")
        global_data = load_dataset(data_args.dataset_reader, url=data_args.dataset_url)

    if data_args.sample:
        for split in global_data.keys():
            global_data[split] = global_data[split]\
                .select([x for x in range(0, data_args.sample)])

    if not data_args.translate_train:
        del global_data["train"]
    if not data_args.translate_validation:
        del global_data["validation"]
    if not data_args.translate_test:
        del global_data["test"]

    def estimate_max_value(data, step=100):
        SelectionsContainer = namedtuple("Selections", ["selections", "max_value"])
        selections = {}
        for key in data.keys():
            selections[key] = [x for x in range(0, len(data[key]))]
            selections[key] = [selections[key][i:i+step] for i in range(0, len(selections[key]), step)]
        max_value = max([ len(selections[x]) for x in selections.keys()])
        return SelectionsContainer(selections=selections, max_value=max_value)

    def generate_slices(data, step=100):
        selections, max_value = estimate_max_value(data, step=step)
        for idx in range(0, max_value):
            new_data = copy.deepcopy(data)
            for key in selections.keys():
                try:
                    selection = selections[key][idx]
                    new_data[key] = new_data[key].select(selection)
                except IndexError:
                    selection = None
                    del new_data[key]
            yield new_data

    def save_dataset_chunk(data, destination):
        
        if os.path.isdir(destination):
            old_data = datasets.load_from_disk(destination, keep_in_memory=True)
            shutil.rmtree(destination, ignore_errors=True)
            for key in old_data.keys():
                if data.get(key, None):
                    data[key] = datasets.concatenate_datasets([old_data[key], data[key]])
                else:
                    data[key] = old_data[key]
        
        data.save_to_disk(destination)

    generator_length = estimate_max_value(
        global_data, 
        step=data_args.translation_generator_batch_size).max_value

    for idx, data in enumerate(generate_slices(
        global_data,
        step=data_args.translation_generator_batch_size)):

        now = str(datetime.datetime.now())
        logger.info("%s", {"step": idx , "total": generator_length, "date": now})

        # Translate the dataset as-is
        data = model.translate(
            data,
            content_field=data_args.content_field,
            output_field=data_args.translation_field,
            **translation_kwargs
        )

        # Get hashtag dictionary
        _, hashtag_dict = model.segment_dataset(
            data,
            content_field=data_args.content_field,
            segmented_content_field=None,
            dictionary=main_hashtag_dict,
            **ws_kwargs
        )

        # Translate the hashtag dictionary
        hashtag_dict = model.translate(
            hashtag_dict,
            **translation_kwargs
        )

        # Replace hashtags with their translations
        data, _ = model.segment_dataset(
            data,
            content_field=data_args.content_field,
            segmented_content_field=data_args.content_replaced_hashtags_field,
            dictionary=hashtag_dict,
            produce_hashtags=True,
            **ws_kwargs
        )

        # Translate the modified tweets
        data = model.translate(
            data,
            content_field=data_args.content_replaced_hashtags_field,
            output_field=data_args.translation_replaced_hashtags_field,
            **translation_kwargs
        )

        inverted_hashtag_dict = {
            "#" + v.replace(" ", ""):v for v in list(hashtag_dict.values())
        }

        data, _ = model.segment_dataset(
            data,
            content_field=data_args.translation_replaced_hashtags_field,
            segmented_content_field=data_args.translation_segmented_hashtags_field,
            dictionary=inverted_hashtag_dict,
            **ws_kwargs
        )

        if data_args.dataset_save_path:
            save_dataset_chunk(data, data_args.dataset_save_path)
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
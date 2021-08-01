from datasets import load_dataset
import logging
import datasets
import json
import sys
import os

from pythonjsonlogger import jsonlogger

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
        "use_encoder": ws_args.use_encoder
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
        decoder_model_name_or_path=ws_args.decoder_model_name_or_path,
        decoder_model_type=ws_args.decoder_model_type,
        decoder_device=ws_args.decoder_device,
        decoder_gpu_batch_size=ws_args.decoder_gpu_batch_size,
        encoder_model_name_or_path=ws_args.encoder_model_name_or_path,
        encoder_model_type=ws_args.encoder_model_type,
        spacy_model=ws_args.spacy_model        
    )

    if data_args.dataset_load_path:
        logger.info("Loading dataset from disk.")
        data = datasets.load_from_disk(data_args.dataset_load_path)
    else:
        logger.info("Loading dataset with reader.")
        data = load_dataset(data_args.dataset_reader, url=data_args.dataset_url)

    if data_args.sample:
        for split in data.keys():
            data[split] = data[split]\
                .select([x for x in range(0, data_args.sample)])

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
        data.save_to_disk(data_args.dataset_save_path)


if __name__ == '__main__':
    main()
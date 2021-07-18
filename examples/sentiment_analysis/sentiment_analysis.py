from dataclasses import dataclass, field
import logging
import os 
import sys

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

logger = logging.getLogger(__name__)

@dataclass
class DataArguments:

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

@dataclass
class TextClassificationArguments:

    sentiment_model: str = field(
        default="finiteautomata/beto-sentiment-analysis"
    )

    batch_size: int = field(
        default=1
    )

    metrics: str = field(
        default="./sentiment_metrics.py"
    )

@dataclass
class WordSegmenterArguments:

    decoder_model_name_or_path: str = field(
        default="gpt2"
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
        default="bert-base-uncased"
    )

    encoder_model_type: str = field(
        default="bert"
    )

    spacy_model: str = field(
        default="en_core_web_sm"
    )

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

    logger.setLevel('DEBUG')
    datasets.utils.logging.set_verbosity('DEBUG')
    transformers.utils.logging.set_verbosity('DEBUG')
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    data = load_dataset(data_args.dataset_reader, url=data_args.dataset_url)[data_args.split]

    model = AutoModelForSequenceClassification.from_pretrained(class_args.sentiment_model)
    tokenizer = AutoTokenizer.from_pretrained(class_args.sentiment_model)
    classifier = pipeline("sentiment-analysis", 
        model=model,
        tokenizer=tokenizer)
    
    sentences = [x[data_args.content_field] for x in data]
    gold = [x[data_args.label_field] for x in data]
    step = class_args.batch_size
    chunks = [ sentences[i:i+step] for i in range(0, len(sentences), step)]
    labels = []
    for chunk in chunks:
        chunk_results = classifier(chunk)
        chunk_labels = [x["label"] for x in chunk_results]
        labels.extend(chunk_labels)
    
    # hashtag_truth_value = ["#" in x for x in sentences]

    metric = load_metric(class_args.metrics)
    eval_results = metric.compute(
        predictions=labels, 
        references=gold)

    logger.info("%s", eval_results)

if __name__ == '__main__':
    main()
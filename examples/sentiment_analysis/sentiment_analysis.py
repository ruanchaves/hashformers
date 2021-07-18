from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging 
import transformers
import datasets 
from datasets import load_dataset
import os 
import sys
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
        default = 'content'
    )

class TextClassificationArguments:

    sentiment_model: str = field(
        default="finiteautomata/beto-sentiment-analysis"
    )

    batch_size: int = field(
        default=1
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
    step = class_args.batch_size
    chunks = [ sentences[i:i+step] for i in range(0, len(sentences), step)]
    for chunk in chunks:
        results = classifier(chunk)
        labels = [x["label"] for x in results]
        print(labels)


if __name__ == '__main__':
    main()
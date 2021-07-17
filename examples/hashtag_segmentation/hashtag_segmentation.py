from dataclasses import dataclass
from itertools import starmap
from json import encoder
from word_segmentation.beamsearch.algorithm import Beamsearch
from word_segmentation.beamsearch.reranker import Reranker
from word_segmentation.evaluation.utils import evaluate_dictionary
from word_segmentation.ensemble.top2_fusion import top2_ensemble
from word_segmentation.beamsearch.data_structures import enforce_prob_dict
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging 
import transformers
import datasets 
from datasets import load_dataset
logger = logging.getLogger(__name__)
import os 
import sys
from typing import Optional
import pathlib
import pandas as pd

@dataclass
class DataArguments:

    source: str = field(
        default='./Test-BOUN'
    )

    evaluate_top_k: int = field(
        default=10
    )

    sample: Optional[int] = field(
        default=None
    )

    output_dir: str = field(
        default='./output'
    )

    save_to_output_dir: bool = field(
        default=True
    )

    load_cache_from_output_dir: bool = field(
        default=False
    )

    decoder_results_filename: str = field(
        default="decoder.csv"
    )

    encoder_results_filename: str = field(
        default="encoder.csv"
    )

    ensemble_results_filename: str = field(
        default="ensemble.csv"
    )


@dataclass
class BeamsearchArguments:
    
    decoder_model_name_or_path: str = field(
        default='gpt2-large'
    )

    decoder_model_type: str = field(
        default='gpt2'
    )

    decoder_device: str = field(
        default='cuda'
    )

    decoder_gpu_batch_size: int = field(
        default=1
    )

    topk: int = field(
        default=20
    )

    steps: int = field(
        default=13
    )

@dataclass
class RerankerArguments:

    encoder_model_name_or_path: str = field(
        default="bert-large-uncased-whole-word-masking",
    )

    encoder_model_type: str = field(
        default="bert"
    )

@dataclass
class EnsembleArguments:

    alpha: float = field(
        default=0.222
    )

    beta: float = field(
        default=0.111
    )

def main():
    parser = HfArgumentParser((BeamsearchArguments, RerankerArguments, EnsembleArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        beamsearch_args, reranker_args, ensemble_args, data_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        beamsearch_args, reranker_args, ensemble_args, data_args = \
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

    # Paths
    if data_args.save_to_output_dir:
        pathlib\
            .Path(data_args.output_dir)\
            .mkdir(parents=True, exist_ok=True)

        encoder_results_path = \
            os.path.join(
                data_args.output_dir,
                data_args.encoder_results_filename
            )

        decoder_results_path = \
            os.path.join(
                data_args.output_dir,
                data_args.decoder_results_filename
            )

        ensemble_results_path = \
            os.path.join(
                data_args.output_dir,
                data_args.ensemble_results_filename
            )

    dataset = load_dataset('text', data_files={'test': data_args.source})
    gold = dataset['test'].to_dict()['text']
    gold = [ x.strip() for x in gold ]
    hashtags = [x.replace(" ", "") for x in gold]

    if data_args.sample:
        gold = gold[0:data_args.sample]
        hashtags = hashtags[0:data_args.sample]



    if data_args.load_cache_from_output_dir and \
        os.path.isfile(decoder_results_path):
        gpt2_run = pd.read_csv(decoder_results_path)
    else:
        gpt2_model = Beamsearch(
            model_name_or_path=beamsearch_args.decoder_model_name_or_path,
            model_type=beamsearch_args.decoder_model_type,
            device=beamsearch_args.decoder_device,
            gpu_batch_size=beamsearch_args.decoder_gpu_batch_size
        )

        gpt2_run = gpt2_model.run(
            hashtags,
            topk=beamsearch_args.topk,
            steps=beamsearch_args.steps
        )

        if data_args.save_to_output_dir:
            gpt2_run.to_csv(decoder_results_path)

    gpt2_metrics = evaluate_dictionary(
        gpt2_run,
        gold,
        n=data_args.evaluate_top_k
    )

    if data_args.load_cache_from_output_dir and \
        os.path.isfile(encoder_results_path):
        bert_run = pd.read_csv(encoder_results_path)
    else:
        bert_model = Reranker(
            model_name_or_path=reranker_args.encoder_model_name_or_path,
            model_type=reranker_args.encoder_model_type
        )

        bert_run = bert_model.rerank(gpt2_run)

        if data_args.save_to_output_dir:
            bert_run_prob_dict = \
                enforce_prob_dict(bert_run)
            bert_run_prob_dict\
                .to_csv(encoder_results_path)

    if data_args.load_cache_from_output_dir and \
        os.path.isfile(ensemble_results_path):
        ensemble = pd.read_csv(ensemble_results_path)
    else:
        ensemble = top2_ensemble(
            bert_run,
            gpt2_run,
            alpha=ensemble_args.alpha,
            beta=ensemble_args.beta
        )

        if data_args.save_to_output_dir:
            ensemble.to_csv(ensemble_results_path)

    ensemble = enforce_prob_dict(
        ensemble,
        score_field="ensemble_rank")

    ensemble_metrics = evaluate_dictionary(
        ensemble,
        gold,
        n=2
    )

    logger.info("Beamsearch metrics:")
    logger.info("%s", gpt2_metrics)

    logger.info("Ensemble metrics:")
    logger.info("%s", ensemble_metrics)

if __name__ == '__main__':
    main()
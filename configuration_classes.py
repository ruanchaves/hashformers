from dataclasses import dataclass, field
from typing import Optional
import json 

def parameters_to_string(*args):
    parameters = [ { item.__class__.__name__ : vars(item) } \
        for item in args ]
    parameters = json.dumps(parameters, indent=4, sort_keys=True)
    return parameters

@dataclass
class ModelArguments:

    model_name_or_path: str = field(
        default='gpt2',
        metadata={
            "help": "The model checkpoint."
        },
    )

    model_type: str = field(
        default='gpt2',
                metadata={
            "help": "Either 'bert' or 'gpt2'."
        }
    )
    
    gpu_batch_size: int = field(
        default=2,
        metadata={"help": "GPU batch size for candidate evaluation."},
    )

@dataclass
class DataEvaluationArguments:
    """
    Arguments pertaining to what data we are going to input our model for evaluation.
    """

    eval_data_file: str = field(
        default=None,
        metadata={"help": "The evaluation data file."},
    )

    eval_dataset_format: Optional[str] = field(
        default="default",
        metadata={"help": "The evaluation data format."},
    )

    expansions_file: str = field(
        default="expansions.json"
    )

    dict_file: str = field(
        default="dict.json"
    )

    report_file: str = field(
        default="report.json"
    )

    validation_expansions_file: str = field(
        default="expansions_validation.json"
    )

    validation_dict_file: str = field(
        default="dict_validation.json"
    )

    validation_report_file: str = field(
        default="report_validation.json"
    )

    index: Optional[int] = field(
        default=-1
    )

    n_chunks: Optional[int] = field(
        default=-1
    )

    logfile: str = field(
        default='model.log'
    )

@dataclass
class BeamsearchArguments:

    topk: int = field(
        default=10,
        metadata={"help": "Beam size."},
    )

    steps: int = field(
        default=2,
        metadata={"help": "Tree depth (maximum amount of words - 1)."},
    )

@dataclass
class RankingArguments:

    topn: int = field(
        default=4,
        metadata={"help": "Top n selected hypothesis."},
    )

    gpu_expansion_batch_size: int = field(
        default=2,
        metadata={"help": "GPU batch size."},
    )

@dataclass
class CNNArguments:

    filter_sizes: str = field(
        default='3,4,5'
    )

    output_dim: int = field(
        default=1
    )

    dropout: float = field(
        default=0.5
    )

    cnn_device: str = field(
        default='cuda'
    )

    token_embedding_size: int = field(
        default=768
    )

    n_filters: int = field(
        default=768
    )

    cnn_learning_rate: float = field(
        default=0.001
    )

    cnn_training_epochs: int = field(
        default=1
    )

    cnn_save_path: str = field(
            default='cnn_model.pth'
    )

    cnn_missed_epoch_limit: int = field(
        default=10
    )

@dataclass
class EncoderArguments:

    compile_generations: bool = field(
        default=True
    )

    compile_pairs: bool = field(
        default=True
    )

    pairs_file: str = field(
        default='pairs.csv'
    )

    generations_file: str = field(
        default='generations.csv'
    )

@dataclass
class MLPArguments:

    mlp_device: str = field(
        default='cuda'
    )

    mlp_training_epochs: int = field(
        default=1
    )

    mlp_save_path: str = field(
        default='mlp_model.pth'
    )

    mlp_learning_rate: float = field(
        default=0.001
    )

    mlp_missed_epoch_limit: int = field(
        default=10
    )

@dataclass
class ModelEvaluationArguments:
    model_evaluation_save_path: str = field(
        default='output/evaluation.csv'
    )
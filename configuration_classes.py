from dataclasses import dataclass, field
from typing import Optional
import json 

def parameters_to_string(*args):
    parameters = [ { item.__class__.__name__ : vars(item) } \
        for item in args ]
    parameters = json.dumps(parameters, indent=4, sort_keys=True)
    return parameters

def parameters_to_command(program, *args):
    parameters = []
    for item in args:
        parameters += ['--{0}={1}'.format(k,v) for k,v in vars(item).items() ]
    cmd = ['python', program] + parameters
    return cmd

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
class ModelEvaluationArguments:
    model_evaluation_save_path: str = field(
        default='output/evaluation.csv'
    )

@dataclass
class BeamsearchManagerArguments:

    expected_worker_load: str = field(
        default=3.0e+9
    )
from dataclasses import dataclass, field
from typing import Optional

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

    index: Optional[int] = field(
        default=-1
    )

    n_chunks: Optional[int] = field(
        default=-1
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

    dropout: int = field(
        default=0.5
    )

    device: str = field(
        default='cuda'
    )

    dropout: int = field(
        default=0.5
    )

    output_dim: int = field(
        default=1
    )

    token_embedding_size: int = field(
        default=768
    )

    n_filters: int = field(
        default=768
    )

    learning_rate: float = field(
        default=0.001
    )

    cnn_training_epochs: int = field(
        default=2
    )

    cnn_save_path: str = field(
            default='cnn_model.pth'
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
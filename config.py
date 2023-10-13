import argparse
from dataclasses import dataclass, field
from typing import Optional


def define_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/mcqa_2",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="piqa",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--setting", type=str, choices=["naive", "random", "diverse", "similar"], default="naive"
    )
    parser.add_argument("--num_examples", type=int, default=5)

    args = parser.parse_args()  # pylint: disable=redefined-outer-name
    return args


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(metadata={"help": "data directory"})
    data_name: str = field(metadata={"help": "data name"})
    setting: str = field(metadata={"help": "setting name"})
    # num_examples: int = field(default=-1, metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    start: int = field(
        default=0,
        metadata={"help": "Start position of sampling"},
    )
    end: int = field(
        default=10,
        metadata={"help": "End position of sampling"},
    )

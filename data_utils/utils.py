import argparse
from enum import Enum
import json
import os
from typing import Any, Dict, List, Tuple, Union


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def read_file(filepath: str) -> Tuple[Any, str]:
    _, ext = os.path.splitext(filepath)
    if ext == ".json":
        return read_json(filepath), ".json"
    elif ext in [".jsonl", ".jl"]:
        return read_jsonl(filepath), ".jsonl"
    elif ext == ".txt":
        return read_txt(filepath), ".txt"
    else:
        raise TypeError(f"Given filetype ({filepath}) is not supported!")


def read_json(filepath: str) -> Dict[Any, Any]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(filepath: str) -> List[Dict[Any, Any]]:
    json_lines = []
    with open(filepath, encoding="utf-8") as f:
        file_lines = f.read().strip().rsplit("\n")
        for line in file_lines:
            json_lines.append(json.loads(line))
    return json_lines


def read_txt(filepath: str) -> List[str]:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().rstrip().split("\n")


def input_file_validator(valid_extensions: Union[Tuple[str, str, str], Tuple[str]]) -> Any:
    def extension(filename: str) -> str:
        if not filename.lower().endswith(valid_extensions):
            raise argparse.ArgumentTypeError("Not a valid filename extension")
        return filename

    return extension


def valid_filetype(arg_input_file: str) -> str:
    _, ext = os.path.splitext(arg_input_file)
    if ext in [".txt", ".jsonl", ".jl"]:
        return arg_input_file
    else:
        msg = f"Given filetype ({arg_input_file}) is not valid! Expected filetype: .txt / .jsonl / .jl"
        raise argparse.ArgumentTypeError(msg)


def write_jsonl(data: List[Dict[Any, Any]], filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        for datum in data:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")

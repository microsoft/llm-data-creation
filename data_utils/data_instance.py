from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class InputExample:
    question: str
    context: str  # QA, MCQA
    endings: Optional[List[str]]  # MCQA
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]

import logging
from typing import Any, List

from torch.utils.data.dataset import Dataset
import tqdm
from transformers import PreTrainedTokenizer
from transformers.tokenization_utils_base import TruncationStrategy

from data_utils.data_instance import InputExample, InputFeatures

logger = logging.getLogger(__name__)


class MCQADataset(Dataset):
    features: List[InputFeatures]

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, examples: List[Any]):
        self.dataset = []
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        for data in examples:
            self.dataset.append(
                InputExample(
                    question=data["question"],
                    context=data["context"],
                    endings=[x for x in data["options"]],
                    label=data["label"],
                )
            )

        self.features = self.convert_examples_to_features(
            self.dataset,
            self.max_seq_length,
            self.tokenizer,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def convert_examples_to_features(
        self,
        examples: List[InputExample],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
    ) -> List[InputFeatures]:
        features = []
        del_count = 0
        for example in tqdm.tqdm(examples, desc="convert examples to features"):
            choices_inputs = []
            for ending in example.endings:
                text_a = example.question
                text_b = ending
                inputs = tokenizer(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    truncation_strategy=TruncationStrategy.ONLY_FIRST,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
                )
                if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                    pass
                choices_inputs.append(inputs)

            if example.label not in example.endings:
                del_count += 1
                continue

            label = example.endings.index(example.label)

            input_ids = [x["input_ids"][0] for x in choices_inputs]
            attention_mask = (
                [x["attention_mask"][0] for x in choices_inputs]
                if "attention_mask" in choices_inputs[0]
                else None
            )
            token_type_ids = (
                [x["token_type_ids"][0] for x in choices_inputs]
                if "token_type_ids" in choices_inputs[0]
                else None
            )

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label,
                )
            )

        for f in features[:2]:
            logger.info("*** Example ***")
            logger.info(f"feature: {f}")
        print(del_count)
        return features


class TCDataset(Dataset):
    features: List[InputFeatures]

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_length: int, examples: List[Any]):
        self.dataset = []
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        for data in examples:
            self.dataset.append(
                InputExample(
                    question=data["question"],
                    context=data["context"],
                    endings=data["options"],
                    label=data["label"],
                )
            )

        self.features = self.convert_examples_to_features(
            self.dataset,
            self.max_seq_length,
            self.tokenizer,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def convert_examples_to_features(
        self,
        examples: List[InputExample],
        max_length: int,
        tokenizer: PreTrainedTokenizer,
    ) -> List[InputFeatures]:
        features = []
        for ex_index, example in tqdm.tqdm(
            enumerate(examples), desc="convert examples to features"
        ):
            if ex_index % 100 == 0:
                logger.info(f"Writing example {ex_index} of {len(examples)}")

            text_a = example.question
            text_b = example.context

            if text_b is not None:
                inputs = tokenizer(
                    text_a,
                    text_b,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    truncation_strategy=TruncationStrategy.ONLY_FIRST,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
                )
            else:
                inputs = tokenizer(
                    text_a,
                    add_special_tokens=True,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    truncation_strategy=TruncationStrategy.ONLY_FIRST,
                    return_overflowing_tokens=True,
                    return_token_type_ids=True,
                )

            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                pass

            if isinstance(example.label, bool):
                if example.label:
                    label = example.endings.index("True")
                else:
                    label = example.endings.index("False")
            else:
                label = example.endings.index(example.label)

            input_ids = inputs["input_ids"][0]
            attention_mask = inputs["attention_mask"][0]
            token_type_ids = inputs["token_type_ids"][0]

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label,
                )
            )

        for f in features[:2]:
            logger.info("*** Example ***")
            logger.info(f"feature: {f}")

        return features

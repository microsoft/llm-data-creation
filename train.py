import json
import logging
import os
from typing import Dict

import numpy as np
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config import DataTrainingArguments, ModelArguments
from data_utils.loader import MCQADataset, TCDataset
from data_utils.reader import Reader
from data_utils.utils import Split

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    (  # pylint: disable=unbalanced-tuple-unpacking
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    if "mcqa" in data_args.data_dir:
        num_labels = int(data_args.data_dir.split("_")[1])
    else:
        num_labels = 2

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    train_examples = Reader(
        data_dir=data_args.data_dir,
        data_name=data_args.data_name,
        mode=Split.TRAIN,
        setting=data_args.setting,
        start=data_args.start,
        end=data_args.end,
    ).load()
    print(data_args.start, data_args.end, len(train_examples))
    dev_examples = Reader(
        data_dir=data_args.data_dir,
        data_name=data_args.data_name,
        mode=Split.VAL,
        setting=data_args.setting,
    ).load()
    test_examples = Reader(
        data_dir=data_args.data_dir,
        data_name=data_args.data_name,
        mode=Split.TEST,
        setting=data_args.setting,
    ).load()

    if "mcqa" in data_args.data_dir:
        model = AutoModelForMultipleChoice.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        train_dataset = (
            MCQADataset(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
            )
            if training_args.do_train
            else None
        )
        val_dataset = (
            MCQADataset(
                examples=dev_examples,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
            )
            if training_args.do_eval
            else None
        )
        test_dataset = (
            MCQADataset(
                examples=test_examples,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
            )
            if training_args.do_predict
            else None
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        train_dataset = (
            TCDataset(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
            )
            if training_args.do_train
            else None
        )
        val_dataset = (
            TCDataset(
                examples=dev_examples,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
            )
            if training_args.do_eval
            else None
        )
        test_dataset = (
            TCDataset(
                examples=test_examples,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
            )
            if training_args.do_predict
            else None
        )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        # import pdb; pdb.set_trace()
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate (Validation) ***")
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w", encoding="utf-8") as writer:
                logger.info(f"saving test results to {output_eval_file}")
                json.dump(obj=result, fp=writer, indent=4)
            logger.info(f"***** Eval results for {output_eval_file} *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
            results.update(result)

    if training_args.do_predict:
        logger.info("*** Evaluate (Prediction) ***")
        # trainer.compute_metrics = None  # Prediciton no need for eval.
        pred_result = trainer.predict(test_dataset=test_dataset)
        pred_result = pred_result.metrics
        output_pred_file = os.path.join(training_args.output_dir, "pred_results.txt")
        if trainer.is_world_process_zero():
            with open(output_pred_file, "w", encoding="utf-8") as writer:
                logger.info(f"saving test results to {output_pred_file}")
                json.dump(obj=pred_result, fp=writer, indent=4)
            logger.info(f"***** Test results for {output_pred_file} *****")
            for key, value in pred_result.items():
                logger.info(f"  {key} = {value}")
            results.update(pred_result)

    return results


def _mp_fn(index):  # pylint: disable=unused-argument
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

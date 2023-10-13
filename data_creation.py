import json
import random

import openai
from sentence_transformers import SentenceTransformer, util
import torch

from config import define_args
from data_utils.reader import Reader
from data_utils.utils import Split
from openai_utils import api_query, read_json
from prompt import example_instruction, system_instruction

if __name__ == "__main__":
    args = define_args()

    random.seed(args.seed)
    data = Reader(
        data_dir=args.data_dir, data_name=args.data_name, mode=Split.TRAIN, setting="naive"
    ).load()

    openai_config = read_json("openai_config.json")
    openai.api_key = openai_config["openai_api_key"]
    output_dir = f"{args.data_dir}/{args.data_name}/train_{args.setting}.jsonl"

    if args.setting in ["diverse", "similar"]:
        sbert_model = SentenceTransformer("all-mpnet-base-v2")

    len_train = len(data)
    print(f"Train Data: {len(data)}")

    example = random.choice(data)

    if "mcqa" in args.data_dir:
        prompt_type = "variant"
    else:
        prompt_type = "fix"

    system_inst = system_instruction(num_examples=args.num_examples, prompt_type=prompt_type)

    example_inst = example_instruction(
        label_space=example["options"],
        question=example["question"],
        answer=example["label"],
        context=example["context"],
        prompt_type=prompt_type,
    )

    print(f"Save in: {output_dir}")
    print(system_inst)

    write_file = open(output_dir, "w", encoding="utf-8")

    history = set()
    count = 0
    total_usage = 0
    while count < len_train:
        print(example_inst)
        text_result, usage = api_query(
            openai=openai, description=system_inst, text=example_inst, model=args.model
        )
        total_usage += usage
        output = text_result.strip().split("\n\n")
        temp_dataset = []
        try:
            assert len(output) == args.num_examples
            for x in output:
                x = x.strip()
                if x[0] != "{":
                    x = "{" + x
                if x[-1] != "}":
                    x = x + "}"
                json_instance = json.loads(x)

                if len(json_instance["Options"]) != len(example["options"]):
                    continue
                else:
                    if prompt_type == "fix" and json_instance["Options"] != example["options"]:
                        continue

                if json_instance["Question"] in history:
                    continue

                if json_instance["Answer"] not in json_instance["Options"]:
                    continue

                history.add(json_instance["Question"])

                data_instance = {
                    "question": json_instance["Question"],
                    "context": json_instance["Context"] if "Context" in json_instance else None,
                    "options": json_instance["Options"],
                    "label": json_instance["Answer"],
                }

                temp_dataset.append(data_instance)

            for x in temp_dataset:
                write_file.write(json.dumps(x, ensure_ascii=False) + "\n")
                count += 1

            print(count)

            if args.setting == "naive":
                pass
            elif args.setting == "random":
                example = random.choice(temp_dataset)
                example_inst = example_instruction(
                    label_space=example["options"],
                    question=example["question"],
                    answer=example["label"],
                    context=example["context"],
                    prompt_type=prompt_type,
                )
            elif args.setting in ["diverse", "similar"]:
                question_query = example["question"]
                question_candidates = [x["question"] for x in temp_dataset]
                query_embedding = sbert_model.encode(question_query, convert_to_tensor=True)
                candidates_embedding = sbert_model.encode(
                    question_candidates, convert_to_tensor=True
                )
                cos_scores = util.cos_sim(query_embedding, candidates_embedding)[0]

                if args.setting == "diverse":
                    _, indices = torch.sort(cos_scores)
                if args.setting == "similar":
                    _, indices = torch.sort(cos_scores, descending=True)

                example = temp_dataset[indices[0].item()]
                example_inst = example_instruction(
                    label_space=example["options"],
                    question=example["question"],
                    answer=example["label"],
                    context=example["context"],
                    prompt_type=prompt_type,
                )
            else:
                raise ValueError("Setting is not valid.")

        except Exception as e:  # pylint: disable=broad-exception-caught
            print(e)

    write_file = open(output_dir.split(".")[0] + "_usage.txt", "w", encoding="utf-8")
    write_file.write(str(total_usage))
    write_file.write("\n")
    write_file.write(str(total_usage / 1000 * 0.002))

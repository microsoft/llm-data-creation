# Making Large Language Models Better Data Creators
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](http://makeapullrequest.com)
[![arXiv](https://img.shields.io/badge/arXiv-2110.08454-b31b1b.svg)]()

This repo provides the model, code & data of our paper: [Making Large Language Models Better Data Creators]() (EMNLP 2023).
[[PDF]]()

## Overview
An LLM-based data creation framework that requires only a single formatting exemplar (e.g., Multiple-choice QA, Open-book QA, Closed-book QA).
Iterative data creation process ensures the development of robust training data for certain downstream task, facilitating the training of powerful small models.

## Table of contents

1. [Setup](#setup)
2. [Hyperparameter](#hyperparameter)
3. [Running](#running)

   3.1. [Data Creation](#data-creation)

   3.2. [Fine-tune Smaller Model](#fine-tune-smaller-model)

<hr/>

## Setup

1. _*(Optional)*_ Create and activate your conda/virtual environment

2. Run `pip install -r requirements.txt`

3. _*(Optional)*_ Add support for CUDA.

4. **Important** Make sure put your OpenAI API key into `openai_config.json`.

<hr/>

## Hyperparameter

| Hyperparameter | Description                                                                         |
|----------------|-------------------------------------------------------------------------------------|
| `data_dir`     | Data directory                                                                      |
| `data_name`    | Data name                                                                           |
| `num_examples` | Number of examples to generate per each iteration                                   |
| `seed`         | Random seed to randomly pick a single formatting example from the train dataset     |
| `setting`      | `naive`, `random`, `diverse`, `similar` (You can run `tree` with different script.) |

<hr/>

## Data folder structure

```
- data
    - mcqa_2
        - piqa
        - winogrande
    - mcqa_5
        - csqa
        - riddle_sense
    - yesno_close
        - boolq
        - creak
        - strategyqa
    - yesno_open
        - bioasq
        - boolq
        - pubmedqa
```

- Possible values for `data_dir`: `data/mcqa_2`, `data/mcqa_5`, `data/yesno_close`, `data/yesno_open`
- Possible values for `data_name`: `piqa`, `winogrande`, `csqa`, `riddle_sense`, `boolq`, `creak`, `strategyqa`, `bioasq`, `pubmedqa`

## Running

### Data Creation
- Setting: `naive`, `random`, `diverse`, `similar`
    ```
    python data_creation.py \
      --data_dir {data_dir} \
      --data_name {data_name} \
      --num_examples {num_examples} \
      --seed {seed} \
      --setting {setting}
    ```
- Setting: `tree`
    ```
    python data_creation_tree.py \
      --data_dir {data_dir} \
      --data_name {data_name} \
      --num_examples {num_examples} \
      --seed {seed} \
      --setting
    ```
### Fine-tune Smaller Model
After data creation, you can train smaller model as follow:
```bash
./script/train.sh {data_dir} {data_name} {setting} {learning_rate} {output_directory}
```

## Citation
If you find our work helpful, please cite the following:
```bib
@InProceedings{lee2023_llm_data_creation,
  author =  {Lee, Dong-Ho and Pujara, Jay and Sewak, Mohit and White, Ryen W and Jauhar, Sujay Kumar},
  title =   {Making Large Language Models Better Data Creators},
  year =    {2023},  
  booktitle = {The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  url = {https://openreview.net/forum?id=2Rdfdri2oT}
}
```

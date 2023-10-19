# Making Large Language Models Better Data Creators
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-green.svg?style=flat-square)](http://makeapullrequest.com)
[![arXiv](https://img.shields.io/badge/arXiv-2110.08454-b31b1b.svg)]()

This repo provides the model, code & data of our paper: "Making Large Language Models Better Data Creators" (EMNLP 2023).

## Overview
**LLM Data Creation** is the process of using a Large Language Model to generate synthetic data for a downstream application.

Our framework enables data creation with LLMs using only one formatting example (e.g., Multiple-choice QA, Open-book QA, Closed-book QA) as an input. 
The process then generates more data in the same format as the input using an iterative process.

It is used to generate data for training smaller task-specific models, such as linear regressors or neural models, in scenarios where there is a lack of human labeled training data.

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
After data creation, you can train and evaluate the smaller model as follow:
```bash
./script/train.sh {data_dir} {data_name} {setting} {learning_rate} {output_directory}
```

## Evaluation
LLM Data Creation was evaluated on 10 publicly available benchmark datasets, comparing smaller models trained on its generated data against other data generation approaches, and human labeled data. The results show that LLM Data Creation performs even better than human labeled data in cross-domain settings, while maintaining comparable performance on in-domain tasks. More details of the model, evaluation, metrics and findings can be found in our paper: "Making Large Language Models Better Data Creators" (EMNLP 2023)

## Tips
- The choice of input formatting example is left to the user, and it’s choice impacts both the domain and content of created data, since the system bootstraps from this one example to create more data.
- Other settings of the LLM, such as temperature and top_p can also control the outputs of LLM Data Creation. While we set both to 1 in our experiments in order to encourage maximum creativity, smaller values may be appropriate strategies – along with other risk mitigation strategies like prompt guardrails and data post-processing and validation – for ensuring output data quality (at the cost of diversity).

## Risks and Limitations
The potential for generating harmful, false or biased responses using LLM Data Creation are no different than those inherent to the underlying LLM being used to generate the data. 
Users should understand these risks and limitations when using this system to create data for downstream applications. 
Instructing the LLM with guardrails to minimize the risks of generating harmful, false or biased responses, as well as employing post-processing techniques to check, filter and sanitize the data may help mitigate the problems with data created with this system.

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

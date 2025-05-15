# <img src="./img/logo.png" width=50px/>PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization

> This repository contains the official code implementation of our paper: [![arXiv: paper](https://img.shields.io/badge/arXiv-paper-red.svg)](https://arxiv.org/abs/xxx)

![PIG](./img/PIG.png)

## Setup

First, create a virtual environment using Anaconda:

```python
conda create -n pig python=3.9.19
conda activate pig
```

Second, you need to install the necessary dependencies:

```python
pip install -r requirements.txt
```

## Datasets

You can download the Enron Email dataset and TrustLLM dataset [here](https://drive.google.com/drive/folders/16Th72F_QcxRAryOIk9L2t0oIps1xnHGW) and place them under the `./data` directory.

## Usage

You can run a privacy jailbreak attack using the following steps:

1. First, modify parameters such as `dataset`, `target_model_name`, `attack_model_name`, or `eval_model_name` in script `run.sh`.
2. Then, execute the privacy jailbreak attack by running `bash run.sh`. Use the `tail` command to monitor the `log` file in real time.
3. Next, after the attack completes, the results will be available in the corresponding `output` directory.
4. Finally, evaluate the results using `python eval.py` to compute various metrics such as the Attack Success Rate (ASR).

## Acknowledgements

Our PIG framework is based on [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak). We thank the team for their open-source implementation.

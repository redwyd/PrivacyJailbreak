# <img src="./img/logo.png" height=30px/> PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization

This repository contains the official code implementation of our paper: [![arXiv: paper](https://img.shields.io/badge/arXiv-paper-red.svg)](https://arxiv.org/abs/2505.09921)

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

## Usage

You can run a privacy jailbreak attack using the following steps:

1. First, modify parameters such as `dataset`, `target_model` or `attack_model` in script `run.sh`.
2. Then, execute the privacy jailbreak attack by running `bash run.sh`.
3. Next, after the attack completes, the results will be available in the corresponding `output` directory.
4. Finally, evaluate the results using `python eval.py` to compute various metrics such as the ASR.

## Acknowledgements

> Our PIG framework is based on [EasyJailbreak](https://github.com/EasyJailbreak/EasyJailbreak). We thank the team for their open-source implementation.

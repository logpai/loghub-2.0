# LogPPT

Repository for the paper: Log Parsing with Prompt-based Few-shot Learning

Thanks for their contribution.

## Requirements
### Library
1. Python 3.8
2. torch
3. transformers
4. ...

To install all library:
```bash
pip install -r requirements.txt
```

We recommend to use conda environment.

```bash
conda create --name LogPPT python=3.8
conda activate LogPPT
pip install -r requirements.txt
```

### 2.2. Pre-trained models
To download the pre-trained language model:
```shell
$ cd pretrained_models/roberta-base
$ bash download.sh
```

## III. Usage:

### Run and evaluate LogPPT on Loghub-2k

```bash
conda activate LogPPT
./run_2k.sh
```

### Run and evaluate LogPPT on LogPub

Please notice you need to download the `full_dataset` of LogPub first.

```bash
conda activate LogPPT
./run_full.sh
```
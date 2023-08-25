# LogPub

LogPub is a collection of large-scale annotated datasets for log parsing based on Loghub, proposed by LogPAI.

Based on LogPub, we propose a more comprehensive benchmark of log parsers. The detailed evaluation results could be found at [RQ_experiments](RQs_experiments/README.md) 🔗.

If you use our benchmark or datasets for research, please cite the following papers:

- Zhihan Jiang, Jinyang Liu, Junjie Huang, Yichen Li, Yintong Huo, Jiazhen Gu, Zhuangbin Chen, Jieming Zhu, Michael R.lyu. [A Large-scale Benchmark for Log Parsing.](https://arxiv.org/abs/2308.10828) Arxiv, 2023. 

In addition, if you use the souce code of our benchmark for research, please also cite the following two papers:

- Khan Zanis Ali, Shin Donghwan, Bianculli Domenico, Briand Lionel. [Guidelines for Assessing the Accuracy of Log Message Template Identification Techniques.](https://dl.acm.org/doi/abs/10.1145/3510003.3510101) ICSE, 2022.

- Jieming Zhu, Shilin He, Jinyang Liu, Pinjia He, Qi Xie, Zibin Zheng, Michael R. Lyu. [Tools and Benchmarks for Automated Log Parsing.](https://arxiv.org/abs/1811.03509) ICSE, 2019.

## Datasets Characteristics

| Software systems          | # Annotated Logs (LogPub) | # Templates  (LogPub) | # Templates (Loghub-2k) |
| ------------------------- | ------------------------- | --------------------- | ----------------------- |
| **Distributed systems**   |                           |                       |                         |
| Hadoop                    | 179,993                   | 236                   | 114                     |
| HDFS                      | 11,167,740                | 46                    | 14                      |
| OpenStack                 | 207,632                   | 48                    | 43                      |
| Spark                     | 16,075,117                | 236                   | 36                      |
| Zookeeper                 | 74,273                    | 89                    | 50                      |
| **Supercomputer systems** |                           |                       |                         |
| BGL                       | 4,631,261                 | 320                   | 120                     |
| HPC                       | 429,987                   | 74                    | 46                      |
| Thunderbird               | 16,601,745                | 1,241                 | 149                     |
| **Operating systems**     |                           |                       |                         |
| Linux                     | 23,921                    | 338                   | 118                     |
| Mac                       | 100,314                   | 626                   | 341                     |
| **Server application**    |                           |                       |                         |
| Apache                    | 51,977                    | 29                    | 6                       |
| OpenSSH                   | 638,946                   | 38                    | 27                      |
| **Standalone software**   |                           |                       |                         |
| HealthApp                 | 212,394                   | 156                   | 75                      |
| Proxifier                 | 21,320                    | 11                    | 8                       |
| **Average**               | **3,601,187**             | **249.1**             | **81.9**                |


## Datasets download

Please first download the full datasets of LogPub via [Zenodo](). (Under organization, will be available ASAP.)

Then, you need to put these datasets into `full_dataset/` following the format of `2k_dataset`.


## Repository Organization 

```
├── 2k_dataset/ # the original Loghub-2k datasets
├── full_dataset/ # unzip the LogPub into this directory
│   └── post_process.py # we provide the heuristic roles used in our annotation of templates 
├── benchmark/
│   ├── evaluation/
│   ├── logparser/
│   ├── old_benchmark/
│   ├── LogPPT/ # contains the modified source code of LogPPT
│   ├── UniParser/ # contains the source code of implemented UniParser
│   ├── run_statistic_2k.sh # the script to run all statistic-based log parsers on Loghub-2k datasets
│   └── run_statistic_full.sh # the script to run all statistic-based log parsers on LogPub datasets
├── result/
│   ├── ...... # 
│   └── ...... # contains the output evaluation metric files and all parsed results
├── RQ_experiments/ # contains the experimental results of RQs
│   ├── RQ1/
│   ├── RQ2/
│   └── RQ3/
├── requirements.txt
└── README.MD
```

## Requirements

Owing to the large scale of the benchmark in the experiments, the requirements of the benchmark of all log parsers are:

- At least 16GB memory.
- At least 100GB storage.
- GPU (for LogPPT and UniParser).

**Installation**

1. Install ```python >= 3.8```
2. ```pip install -r requirements.txt```


## One-Click Results Reproduction

Running the entire benchmark using Logpub datasets requires more than **48 hours** to complete.

Note that if you would like to evaluate your parser, *one can easily put their parsed results following the format as the files shown in `result/`, and run our evluation code.*

## Large-scale benchmarking

If you woud like to re-run all parsers using Logpub, please follow our large-scale benchmarking steps.

### Quick Demo using Drain

We give a demo script to run Drain on both Loghub-2k and LogPub, this will takes about 2-3 hours.

```bash
cd benchmark/
./demo.sh
```

### Evaluation of all 15 parsers

One can follow the steps to evaluate all parsers using Loghub-2k or the proposed Logpub datasets. The overall time cost is more than 48 hours.

- Run all statistic-based log parsers on Loghub-2k

```bash
cd benchmark/
./run_statistic_2k.sh
```

- Run all statistic-based log parsers on LogPub

```bash
cd benchmark/
./run_statistic_full.sh
```

- Run Semantic-based log parsers: LogPPT & UniParser

  Since these methods are quite different with other log parsers, and they requires a GPU to support efficient parsing, we seperate their environments from other log parsers. Please refer to the README file of [LogPPT](benchmark/LogPPT/README.md) or [UniParser](benchmark/UniParser/README.md) to use one-click script to parse and evaluate each log parsers respectively.


## Acknowledgements

This repository is based on the repositories of the following two papers:

- [ICSE'19] Jieming Zhu, Shilin He, Jinyang Liu, Pinjia He, Qi Xie, Zibin Zheng, Michael R. Lyu. Tools and Benchmarks for Automated Log Parsing. International Conference on Software Engineering (ICSE), 2019.

- [ICSE'22] Khan Zanis Ali, Shin Donghwan, Bianculli Domenico, Briand Lionel. Guidelines for Assessing the Accuracy of  Log Message Template Identification Techniques. International Conference on Software Engineering (ICSE), 2022.
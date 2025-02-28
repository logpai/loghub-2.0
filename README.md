# LogLSHD

LogLSHD is an enhanced log parsing framework built upon Loghub-2.0, integrating Locality-Sensitive Hashing (LSH) techniques to improve log template extraction and clustering efficiency.


## Datasets download

Please first download the full datasets of Loghub-2.0 via [Zenodo](https://zenodo.org/record/8275861).

Then, you need to put these datasets into `full_dataset/` following the format of `2k_dataset`.


## Requirements

Owing to the large scale of the benchmark in the experiments, the requirements of the benchmark of all log parsers are:

- At least 16GB memory.
- At least 100GB storage.
- GPU (for LogPPT and UniParser).

**Installation**

1. Install ```python >= 3.8```
2. ```pip install -r requirements.txt```


## One-Click Results Reproduction

Running the entire benchmark using Loghub-2.0 datasets requires more than **48 hours** to complete.

Note that if you would like to evaluate your parser, *one can easily put their parsed results following the format as the files shown in `result/`, and run our evluation code.*



### Evaluation of all 15 parsers

One can follow the steps to evaluate all parsers using Loghub-2k or the proposed Loghub-2.0 datasets. The overall time cost is more than 48 hours.


- Run all statistic-based log parsers on Loghub-2.0

```bash
cd benchmark/
./run_lsh_full.sh
```

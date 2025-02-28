# LogLSHD

LogLSHD is an enhanced log parsing framework built upon [Loghub-2.0](https://github.com/logpai/loghub-2.0), integrating Locality-Sensitive Hashing (LSH) techniques to improve log template extraction and clustering efficiency.

## Getting Started

To use LogLSHD for log parsing, follow these steps:

### Datasets download

Please first download the full datasets of Loghub-2.0 via [Zenodo](https://zenodo.org/record/8275861).

Then, you need to put these datasets into `full_dataset/` following the format of `2k_dataset`.

### Install dependencies

1. Install ```python >= 3.8```
2. ```pip install -r requirements.txt```

### Evaluation of LogLSHD with 14 datasets


```bash
cd benchmark/
./run_lsh_full.sh
```

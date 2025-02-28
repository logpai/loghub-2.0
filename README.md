# LogLSHD

LogLSHD is an enhanced log parsing framework built upon [Loghub-2.0](https://github.com/logpai/loghub-2.0), integrating Locality-Sensitive Hashing (LSH) techniques to improve log template extraction and clustering efficiency.

## Getting Started

To use LogLSHD for log parsing, follow these steps:

### 1. Datasets download

Please first download the full datasets of Loghub-2.0 via [Zenodo](https://zenodo.org/record/8275861).

After downloading, place the datasets in the `full_dataset/` directory, ensuring the format matches the `2k_dataset` directory.

### 2. Install Dependencies

1. Install ```python >= 3.8```
2. ```pip install -r requirements.txt```

### 3. Run Benchmark


- Evaluate LogLSHD performance on 14 datasets:

```bash
cd benchmark/
./run_lsh_full.sh
```

- Run all statistic-based log parsers on Loghub-2k:

```bash

cd benchmark/
./run_statistic_2k.sh
```

- Run all statistic-based log parsers on Loghub-2.0:

```bash
cd benchmark/
./run_statistic_full.sh
```
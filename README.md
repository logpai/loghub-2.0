# LogLSHD

LogLSHD is an enhanced log parsing framework built upon [Loghub-2.0](https://github.com/logpai/loghub-2.0), integrating Locality-Sensitive Hashing (LSH) techniques to improve log template extraction and clustering efficiency.

## Method Overview
LogLSHD consists of the following stages:
1. **Preprocessing**
In this stage, raw log messages are preprocessed to replace common variables (such as IP addresses and domain names) with placeholders using domain-specific regular expressions.
2. **Initial Log Grouping**
This stage employs simple heuristic methods to initially group unstructured log messages. It assesses whether logs belong to the same group by comparing the token count, content length, and specific position characters.
3. **Merging Initial Groups into Log Clusters** 
After initial grouping, a stricter similarity comparison is applied to aggregate the grouped logs. Using a Jaccard similarity threshold, similar log messages are clustered together for template extraction.
4. **Template Extraction**
Finally, the method randomly selects a representative sample of 10 logs from each group for template extraction, resulting in the formation of final log templates. The template extraction process leverages Dynamic Time Warping (DTW) to optimize accuracy in conjunction with similarity grouping.

![LogLSHD](https://imgur.com/a/9KufE4C.jpg "Structure of LogLSHD.")

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

## Default Jaccard Thresholds of LSH for Different Datasets

The following table presents the default Jaccard similarity thresholds used in LogLSHD for various datasets.

| Dataset     | Jaccard Threshold |
|------------|------------------|
| Proxifier  | 1.00             |
| Linux      | 0.65             |
| Apache     | 0.65             |
| Zookeeper  | 0.80             |
| Hadoop     | 0.85             |
| HealthApp  | 0.65             |
| OpenStack  | 0.70             |
| HPC        | 0.70             |
| Mac        | 0.95             |
| OpenSSH    | 0.85             |
| Spark      | 0.90             |
| Thunderbird| 0.60             |
| BGL        | 0.90             |
| HDFS       | 0.80             |

## Research Paper

For more details, please refer to the paper: [LogLSHD: Locality-Sensitive Hashing for Accurate and Efficient Log Parsing](https://arxiv.org/abs/2504.02172).
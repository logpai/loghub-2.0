## Requirements
Python 3.8.0

```shell
conda create --name UniParser python=3.8
conda activate UniParser
pip install -r requirements.txt
```

### Step 0: Download Glove Embedding

Download from this link () and put it in the correct directory

```shell
wget https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/gluon/embeddings/glove/glove.6B.zip
```

Unzip the file and add a line in the first line of the embedding file as follows, to transform it from glove format into word2vec format

```
400000 50
```

Here 400000 is the number of lines of the embedding file, and 50 is the embedding dimension.

### Run and evaluate UniParser on Loghub-2k

```bash
conda activate UniParser
./run_2k.sh
```

### Run and evaluate UniParser on LogPub

Please notice you need to download the `full_dataset` of LogPub first.

```bash
conda activate UniParser
./run_full.sh
```

## Evaluation of LSTM-based log parsers (UniParser)

Since the performances of LSTM-based log parsers are significantly influenced by the delimiters, and the authors of UniParser do not provide the detailed delimiters. Therefore, for UniParser, we only use space as delimiter, and temporarily change the evaluation of Parsing Accuracy: (We check each token after tokenize, and any token found to contain <*> is consequently deemed as <*>).
This is an imprecise estimation of PA, yet it ensures the general trend is accurately represented.
Others employing UniParser may leverage other delimiters to get a more precise measurement of performance.



<!-- ### Step 1: Preprocess VALB dataset 

Transfer the VALB format into the format that can be used by the model.

```bash
conda activate UniParser
python process_log_parsing_input_to_ner.py
```

### Step 2: Train the model
```bash
# train on all datasets
conda activate UniParser
python TrainNERLogAll.py
```

### Step 3: Infer the data with trained models
Input is in a .csv file format, with a column of "Content" 
```bash
# infer on all datasets
conda activate UniParser
python InferNERLogAll.py
```
 -->

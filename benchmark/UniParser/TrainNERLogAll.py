import torch
from torch.optim import Adam
from ner.corpus import Corpus
from ner.models import NERModel
from ner.lr_finder import LRFinder
from ner.trainer import Trainer
from pprint import pprint
import os
import argparse
import csv
import sys
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl
def write_experiment_results(outfile, result_dict):
    # Write the results to a CSV file
    # row_to_save = vars(args)  ## add args
    # row_to_save.update(result)

    with open(outfile, "a", newline="") as csvfile:
        if sys.platform == "win32":
            # Acquire a file write lock on Windows
            msvcrt.locking(csvfile.fileno(), msvcrt.LK_LOCK, 1)
        else:
            # Acquire a file write lock on Linux
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_EX)

        fieldnames = list(result_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If this is the first row, write the header
        if csvfile.tell() == 0:
            writer.writeheader()

        # Write the row for this experiment
        writer.writerow(result_dict)
        print(f"Experiment record saved to {outfile}.")

        if sys.platform == "win32":
            # Release the file write lock on Windows
            msvcrt.locking(csvfile.fileno(), msvcrt.LK_UNLOCK, 1)
        else:
            # Release the file write lock on Linux
            fcntl.flock(csvfile.fileno(), fcntl.LOCK_UN)


def set_seed(seed=42):
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    config = argparse.ArgumentParser()
    config.add_argument('-full', '--full_data',
                        help="Set this if you want to test on full dataset",
                        default=False, action='store_true')
    config = config.parse_args()
    data_type = 'full' if config.full_data else '2k'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    find_lr = False
    set_seed(42)
    use_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = ['Apache', 'BGL', 'HDFS', 'HPC', 'Hadoop', 'HealthApp', 'Linux', 'Mac', 'OpenSSH', 'OpenStack',
             'Proxifier', 'Spark', 'Thunderbird', 'Zookeeper']

    all_results = {}
    for log_file in files:
        training_config = {
            "max_epochs": 100,
            "no_improvement": 10,
            "batch_size": 16,
            "lr": 1e-1,
            "weight_decay": 1e-2

        }
        input_folder = os.path.join(f"{data_type}_annotations", log_file, "logpub_bin_random")
        corpus = Corpus(
            input_folder=input_folder,
            min_word_freq=3,
            batch_size=training_config['batch_size'],
            wv_file="model/glove.6B.50d.txt"
        )
        print(f"Train set: {len(corpus.train_dataset)} sentences")
        print(f"Val set: {len(corpus.val_dataset)} sentences")
        print(f"Test set: {len(corpus.test_dataset)} sentences")
        # configurations building block
        base = {
            "word_input_dim": len(corpus.word_field.vocab),
            "char_pad_idx": corpus.char_pad_idx,
            "word_pad_idx": corpus.word_pad_idx,
            "tag_names": corpus.tag_field.vocab.itos,
            "device": use_device
        }
        w2v = {
            "word_emb_pretrained": corpus.word_field.vocab.vectors if corpus.word_vectors else None,
            "word_emb_dim": 50,
            "word_emb_dropout": 0.5,
            "word_emb_froze": True
        }
        cnn = {
            "use_char_emb": True,
            "char_emb_pretrained": None,
            "char_input_dim": len(corpus.char_field.vocab),
            "char_emb_dim": 37,
            "char_emb_dropout": 0.25,
            "char_cnn_filter_num": 4,
            "char_cnn_kernel_size": 3,
            "char_cnn_dropout": 0.25
        }
        lstm = {
            "lstm_hidden_dim": 64,
            "lstm_layers": 2,
            "lstm_dropout": 0.1
        }
        # attn = {
        #     "attn_heads": 16,
        #     "attn_dropout": 0.25
        # }
        # transformer = {
        #     "model_arch": "transformer",
        #     "trf_layers": 1,
        #     "fc_hidden": 256,
        # }
        configs = {
            # "bilstm": base,
            # "bilstm+w2v": {**base, **w2v},
            "bilstm+w2v+cnn": {**base, **w2v, **cnn, **lstm},
            # "bilstm+w2v+cnn+attn": {**base, **w2v, **cnn, **attn},
            # "transformer+w2v+cnn": {**base, **transformer, **w2v, **cnn, **attn}
        }
        if find_lr:
            suggested_lrs = {}
            for model_name in configs:
                model = NERModel(**configs[model_name])
                lr_finder = LRFinder(model, Adam(model.parameters(), lr=1e-4, weight_decay=1e-2), device=use_device)
                lr_finder.range_test(corpus.train_iter, corpus.val_iter, end_lr=1, num_iter=25, step_mode="exp")
                _, suggested_lrs[model_name] = lr_finder.plot(skip_start=10, skip_end=0)
        else:
            suggested_lrs = {model_name: training_config['lr'] for model_name in configs}
        pprint(suggested_lrs)

        histories = {}
        for model_name in configs:
            if not os.path.exists(f"saved_states_{data_type}/{log_file}"):
                os.makedirs(f"saved_states_{data_type}/{log_file}")
            checkpoint_path = f"saved_states_{data_type}/{log_file}/{model_name}-" \
                              f"w{configs[model_name]['word_emb_dim']}c{configs[model_name]['char_emb_dim']}f{configs[model_name]['char_cnn_filter_num']}k{configs[model_name]['char_cnn_kernel_size']}-" \
                              f"lstm{configs[model_name]['lstm_hidden_dim']}L{configs[model_name]['lstm_layers']}-" \
                              f"lr{suggested_lrs[model_name]}-epoch{training_config['max_epochs']}bz{training_config['batch_size']}.pt"
            print(f"Start Training: {model_name}")
            model = NERModel(**configs[model_name])
            trainer = Trainer(
                model=model,
                data=corpus,
                optimizer=Adam(model.parameters(), lr=suggested_lrs[model_name], weight_decay=training_config['weight_decay']),
                device=use_device,
                checkpoint_path=checkpoint_path
            )
            # histories[model_name] = trainer.train(max_epochs=training_config['max_epochs'], no_improvement=training_config['no_improvement'])
            histories[model_name] = trainer.train_parsing_accuracy(max_epochs=training_config['max_epochs'], no_improvement=training_config['no_improvement'])
            print(f"Done Training: {model_name}")
            print()
            if os.path.exists(checkpoint_path):
                trainer.model.load_state(checkpoint_path)
            else:
                print("No checkpoint found. Use model's last state for inference.")
            # sentence = ' '.join(corpus.test_dataset.examples[0].word)
            # words, infer_tags, unknown_tokens = trainer.infer(sentence=sentence)
        pprint(histories)
        print("==============================")
        print(f"Dataset: {log_file}")
        print(f"Num Param: {histories['bilstm+w2v+cnn']['num_params']}")
        # print(f"Best Val F1 {round(histories['bilstm+w2v+cnn']['best_val_f1']*100, 2)}")
        # print(f"Best Test F1 {round(histories['bilstm+w2v+cnn']['test_f1']*100, 2)}")
        print(f"Best Val EM {round(histories['bilstm+w2v+cnn']['best_val_em']*100, 2)}")
        print(f"Best Test EM {round(histories['bilstm+w2v+cnn']['test_em']*100, 2)}")
        print(f"Model saved to: {checkpoint_path}")
        print("==============================")
        all_results[log_file] = round(histories['bilstm+w2v+cnn']['test_f1']*100, 2)

        write_csv = configs["bilstm+w2v+cnn"].copy()
        write_csv.update(training_config)
        write_csv.update({"Dataset": log_file, "TestEM": round(histories['bilstm+w2v+cnn']['test_em']*100, 2)})
        write_csv.pop("word_emb_pretrained")
        write_experiment_results(f'summary_results_bilstm+w2v+cnn_{data_type}.csv', write_csv)

    print("======================")
    for k, v in all_results.items():
        print(f"{k}\t{v}")
    print("======================")



"""
Experiment record saved to summary_results_bilstm+w2v+cnn.csv.
======================
Apache  99.68
BGL     97.11
HDFS    98.86
HPC     99.94
Hadoop  98.24
HealthApp       99.67
Linux   98.5
Mac     87.53
OpenSSH 98.03
OpenStack       99.3
Proxifier       99.94
Spark   99.97
Thunderbird     95.63
Zookeeper       99.32
======================
"""

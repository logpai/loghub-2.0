"""Automatic label search helpers."""

import itertools
from collections import Counter

import torch
import tqdm
import multiprocessing
import numpy as np
import scipy.spatial as spatial
import scipy.special as special
import scipy.stats as stats

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_initial_label_words(model, loader, val_label=1):
    initial_label_words = []
    for step, batch in enumerate(loader):
        label = batch.pop('ori_labels')
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        logits = torch.topk(outputs.logits.log_softmax(dim=-1), k=5).indices
        for i in range(label.shape[0]):
            for j in range(len(label[i])):
                if label[i][j] != val_label:
                    continue
                initial_label_words.extend(logits[i][j].detach().cpu().clone().tolist())

    return list(set(initial_label_words))


def find_labels(model, train, eval, keywords, val_label=1, k_likely=1000, k_neighbors=None, top_n=-1, vocab=None):
    # Get top indices based on conditional likelihood using the LM.
    model.to(device)
    model.eval()
    initial_label_words = get_initial_label_words(model, train, val_label)
    label_words_freq = {}
    for batch in eval:
        for inp in batch['input_ids'].detach().clone().tolist():
            for i in inp:
                if i in initial_label_words and i not in keywords:
                    if i not in label_words_freq.keys():
                        label_words_freq[i] = 0
                    label_words_freq[i] += 1
    # label_words = [x[0] for x in Counter(label_words_freq).most_common(n=10)]
    label_words_freq = {x[0]: x[1] for x in sorted(label_words_freq.items(), key=lambda k: k[1], reverse=True)}
    return label_words_freq
    # Brute-force search all valid pairings.
    # pairings = list(itertools.product(*label_candidates))
    #
    # if is_regression:
    #     eval_pairing = eval_pairing_corr
    #     metric = "corr"
    # else:
    #     eval_pairing = eval_pairing_acc
    #     metric = "acc"
    #
    # # Score each pairing.
    # pairing_scores = []
    # with multiprocessing.Pool(initializer=init, initargs=(train_logits, train_labels)) as workers:
    #     with tqdm.tqdm(total=len(pairings)) as pbar:
    #         chunksize = max(10, int(len(pairings) / 1000))
    #         for score in workers.imap(eval_pairing, pairings, chunksize=chunksize):
    #             pairing_scores.append(score)
    #             pbar.update()
    #
    # # Take top-n.
    # best_idx = np.argsort(-np.array(pairing_scores))[:top_n]
    # best_scores = [pairing_scores[i] for i in best_idx]
    # best_pairings = [pairings[i] for i in best_idx]
    #
    # logger.info("Automatically searched pairings:")
    # for i, indices in enumerate(best_pairings):
    #     logger.info("\t| %s (%s = %2.2f)", " ".join([vocab[j] for j in indices]), metric, best_scores[i])
    #
    # return best_pairings

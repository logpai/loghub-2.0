import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from spacy.lang.id import Indonesian
import spacy
from sklearn.metrics import f1_score, classification_report


class Trainer(object):

    def __init__(self, model, data, optimizer, device, checkpoint_path=None):
        self.device = device
        self.model = model.to(self.device)
        self.data = data
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path

    @staticmethod
    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def f1_positive(self, preds, y, full_report=False):
        """

                Args:
                    preds: predictions of all examples: a list of labels (label is a list), eg. [[1, 1, 1, 1, 2], [1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 2], ...]
                    y: the same shape with preds: a list of labels
                    full_report:

                Returns:

        """
        index_o = self.data.tag_field.vocab.stoi["O"]
        # print(f"preds: {preds}, y: {y}")
        # take all labels except padding and "O"
        positive_labels = [i for i in range(len(self.data.tag_field.vocab.itos))
                           if i not in (self.data.tag_pad_idx, index_o)]
        # make the prediction one dimensional to follow sklearn f1 score input param
        flatten_preds = [pred for sent_pred in preds for pred in sent_pred]
        # remove prediction for padding and "O"
        positive_preds = [pred for pred in flatten_preds
                          if pred not in (self.data.tag_pad_idx, index_o)]
        # make the true tags one dimensional to follow sklearn f1 score input param
        flatten_y = [tag for sent_tag in y for tag in sent_tag]
        # print(f"positive preds: {positive_preds}; positive y: {positive_labels}")
        # print(f"flatten preds: {flatten_preds}; flatten y: {flatten_y}")
        if full_report:
            # take all names except padding and "O"
            positive_names = [self.data.tag_field.vocab.itos[i]
                              for i in range(len(self.data.tag_field.vocab.itos))
                              if i not in (self.data.tag_pad_idx, index_o)]
            print(classification_report(
                y_true=flatten_y,
                y_pred=flatten_preds,
                labels=positive_labels,
                target_names=positive_names
            ))
        # average "micro" means we take weighted average of the class f1 score
        # weighted based on the number of support
        return f1_score(
            y_true=flatten_y,
            y_pred=flatten_preds,
            labels=positive_labels,
            average="micro"
        ) if len(positive_preds) > 0 else 0

    def em_positive(self, preds, y):
        # index_o = self.data.tag_field.vocab.stoi["O"]

        # make the prediction and remove the pad token
        real_preds = [[pred for pred in sent_pred if pred != self.data.tag_pad_idx] for sent_pred in preds]
        # make the true tags one dimensional to follow sklearn f1 score input param
        real_y = [[tag for tag in sent_tag if tag != self.data.tag_pad_idx] for sent_tag in y]

        score = sum([i==j for i, j in zip(real_preds, real_y)])/len(y)
        return score

    def epoch(self):
        epoch_loss = 0
        true_tags_epoch = []
        pred_tags_epoch = []
        self.model.train()
        for batch in self.data.train_iter:
            # words = [sent len, batch size]
            words = batch.word.to(self.device)
            # chars = [batch size, sent len, char len]
            chars = batch.char.to(self.device)
            # tags = [sent len, batch size]
            true_tags = batch.tag.to(self.device)
            self.optimizer.zero_grad()
            # print(f"shape words {words.shape}, char {chars.shape}, tags {true_tags.shape}")
            # for idx in range(words.shape[1]):
            #     print(words[:, idx].tolist())
            #     print([self.data.word_field.vocab.itos[int(words[j, idx])] for j in range(words.shape[0])])
            #     print([self.data.tag_field.vocab.itos[int(true_tags[j, idx])] for j in range(true_tags.shape[0])])
            pred_tags_list, batch_loss = self.model(words, chars, true_tags)
            pred_tags_epoch += pred_tags_list
            # to calculate the loss and f1, we flatten true tags
            true_tags_epoch += [
                [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                for sent_tag in true_tags.permute(1, 0).tolist()
            ]
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        epoch_score = self.f1_positive(pred_tags_epoch, true_tags_epoch)
        epoch_em = self.em_positive(pred_tags_epoch, true_tags_epoch)
        return epoch_loss / len(self.data.train_iter), epoch_score, epoch_em

    def evaluate(self, iterator, full_report=False):
        epoch_loss = 0
        all_words = []
        true_tags_epoch = []
        pred_tags_epoch = []
        self.model.eval()
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                words = batch.word.to(self.device)
                chars = batch.char.to(self.device)
                true_tags = batch.tag.to(self.device)
                pred_tags, batch_loss = self.model(words, chars, true_tags)
                pred_tags_epoch += pred_tags
                true_tags_epoch += [
                    [tag for tag in sent_tag if tag != self.data.tag_pad_idx]
                    for sent_tag in true_tags.permute(1, 0).tolist()
                ]
                all_words += [[self.data.word_field.vocab.itos[wordid] for wordid in sent] for sent in words]
                epoch_loss += batch_loss.item()
        epoch_score = self.f1_positive(pred_tags_epoch, true_tags_epoch, full_report)
        epoch_em = self.em_positive(pred_tags_epoch, true_tags_epoch)
        # print(f"em positive: {epoch_em}, len: {len(true_tags_epoch)}")
        all_true_tags = [[self.data.tag_field.vocab.itos[t] for t in tags] for tags in true_tags_epoch]
        all_pred_tags = [[self.data.tag_field.vocab.itos[t] for t in tags] for tags in pred_tags_epoch]
        # for i in range(len(all_true_tags)):
        # for i in range(50):
        #     print(f"{i}/{len(all_true_tags)} Words: {all_words[i]} Preds: {all_pred_tags[i]}, Tags: {all_true_tags[i]}")
        return epoch_loss / len(iterator), epoch_score, epoch_em

    def train(self, max_epochs, no_improvement=None):
        history = {
            "num_params": self.model.count_parameters(),
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_f1": [],
        }
        elapsed_train_time = 0
        best_val_f1 = 0
        best_epoch = None
        lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=3,
            factor=0.3,
            mode="max",
            verbose=True
        )
        epoch = 1
        n_stagnant = 0
        stop = False
        while not stop:
            start_time = time.time()
            train_loss, train_f1, train_em = self.epoch()
            end_time = time.time()
            elapsed_train_time += end_time - start_time
            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            val_loss, val_f1, val_em = self.evaluate(self.data.val_iter)
            lr_scheduler.step(val_f1)
            # take the current model if it it at least 1% better than the previous best F1
            if self.checkpoint_path and val_f1 > (best_val_f1 * 1.01):
                print(f"Epoch-{epoch}: found better Val F1: {val_f1:.4f}, saving model...")
                self.model.save_state(self.checkpoint_path)
                best_val_f1 = val_f1
                best_epoch = epoch
                n_stagnant = 0
            else:
                n_stagnant += 1
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            if epoch >= max_epochs:
                print(f"Reach maximum number of epoch: {epoch}, stop training.")
                stop = True
            elif no_improvement is not None and n_stagnant >= no_improvement:
                print(f"No improvement after {n_stagnant} epochs, stop training.")
                stop = True
            else:
                epoch += 1
        if self.checkpoint_path and best_val_f1 > 0:
            self.model.load_state(self.checkpoint_path)
        test_loss, test_f1, test_em = self.evaluate(self.data.test_iter)
        history["best_val_f1"] = best_val_f1
        history["best_epoch"] = best_epoch
        history["test_loss"] = test_loss
        history["test_f1"] = test_f1
        history["elapsed_train_time"] = elapsed_train_time
        return history

    def train_parsing_accuracy(self, max_epochs, no_improvement=None):
        history = {
            "num_params": self.model.count_parameters(),
            "train_loss": [],
            "train_f1": [],
            "train_acc": [],
            "val_loss": [],
            "val_f1": [],
            "val_acc": [],
        }
        elapsed_train_time = 0
        best_val_em = 0
        best_epoch = None
        lr_scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            patience=3,
            factor=0.3,
            mode="max",
            verbose=True
        )
        epoch = 1
        n_stagnant = 0
        stop = False
        while not stop:
            start_time = time.time()
            train_loss, train_f1, train_em = self.epoch()
            end_time = time.time()
            elapsed_train_time += end_time - start_time
            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)
            history["train_acc"].append(train_em)
            val_loss, val_f1, val_em = self.evaluate(self.data.val_iter)
            lr_scheduler.step(val_em)
            # take the current model if it it at least 1% better than the previous best F1
            if self.checkpoint_path and val_em > (best_val_em * 1.01):
                print(f"Epoch-{epoch}: found better Val EM: {val_em:.4f}, saving model...")
                self.model.save_state(self.checkpoint_path)
                best_val_em = val_em
                best_epoch = epoch
                n_stagnant = 0
            else:
                n_stagnant += 1
            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_acc"].append(val_em)
            if epoch >= max_epochs:
                print(f"Reach maximum number of epoch: {epoch}, stop training.")
                stop = True
            elif no_improvement is not None and n_stagnant >= no_improvement:
                print(f"No improvement after {n_stagnant} epochs, stop training.")
                stop = True
            else:
                epoch += 1
        if self.checkpoint_path and best_val_em > 0:
            self.model.load_state(self.checkpoint_path)
        print("Starting Evaluate on test set")
        test_loss, test_f1, test_em = self.evaluate(self.data.test_iter)
        history["best_val_em"] = best_val_em
        history["best_epoch"] = best_epoch
        history["test_loss"] = test_loss
        history["test_f1"] = test_f1
        history["test_em"] = test_em
        history["elapsed_train_time"] = elapsed_train_time
        return history

    # @staticmethod
    # def visualize_attn(tokens, weights):
    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     im = ax.imshow(weights, cmap=plt.get_cmap("gray"))
    #     ax.set_xticks(list(range(len(tokens))))
    #     ax.set_yticks(list(range(len(tokens))))
    #     ax.set_xticklabels(tokens)
    #     ax.set_yticklabels(tokens)
    #     # Create colorbar
    #     _ = ax.figure.colorbar(im, ax=ax)
    #     # Rotate the tick labels and set their alignment.
    #     plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
    #              rotation_mode="anchor")
    #     plt.tight_layout()
    #     plt.show()

    def infer(self, sentence, true_tags=None, do_print=False):
        # print(f"Infering with {self.device}")
        self.model.eval()
        # tokenize sentence
        # nlp = Indonesian()
        # ### To use spacy for tokenization, should first install spacy and download the model
        # ### Download the model: python -m spacy download en_core_web_sm
        # ### Or first download en_core_web_sm-3.0.0-py3-none-any.whl
        # ###      and then pip install /path/to/en_core_web_sm-3.0.0-py3-none-any.whl
        # nlp = spacy.load("en_core_web_sm")
        # tokens = [token.text for token in nlp(sentence)]
        tokens = sentence.split(' ')
        max_word_len = max([len(token) for token in tokens])
        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[token.lower()] for token in tokens]
        char_pad_id, word_pad_idx = self.data.char_pad_idx, self.data.word_pad_idx
        # numericalized_chars = []
        # for token in tokens:
        #     numericalized_chars.append(
        #         [self.data.char_field.vocab.stoi[char] for char in token]
        #         + [char_pad_id for _ in range(max_word_len - len(token))]
        #     )
        numericalized_chars = [[self.data.char_field.vocab.stoi[char] for char in token] + [char_pad_id for _ in range(max_word_len - len(token))] for token in tokens]
        # find unknown words
        unk_idx = self.data.word_field.vocab.stoi[self.data.word_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        # begin prediction
        token_tensor = torch.as_tensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1).to(self.device)
        char_tensor = torch.as_tensor(numericalized_chars)
        char_tensor = char_tensor.unsqueeze(0).to(self.device)
        # print(f"token tensor shape {token_tensor.shape}, char tensor shape {char_tensor.shape}, num tokens {len(tokens)}, max chars {max_word_len}, char pad {char_pad_id}/{self.data.char_field.pad_token}, word pad {self.data.word_pad_idx}/{self.data.word_field.pad_token}")
        predictions, _ = self.model(token_tensor, char_tensor)
        # print(len(predictions), predictions)
        # convert results to tags
        predicted_tags = [self.data.tag_field.vocab.itos[t] for t in predictions[0]]
        # print inferred tags
        max_len_token = max([len(token) for token in tokens] + [len('word')])
        max_len_tag = max([len(tag) for tag in predicted_tags] + [len('pred')])
        if do_print:
            print(
                f"{'word'.ljust(max_len_token)}\t{'unk'.ljust(max_len_token)}\t{'pred tag'.ljust(max_len_tag)}"
                + ("\ttrue tag" if true_tags else "")
            )
            for i, token in enumerate(tokens):
                is_unk = "✓" if token in unks else ""
                print(
                    f"{token.ljust(max_len_token)}\t{is_unk.ljust(max_len_token)}\t{predicted_tags[i].ljust(max_len_tag)}"
                    + (f"\t{true_tags[i]}" if true_tags else "")
                )
        return tokens, predicted_tags, unks

    def infer_batch(self, sentences, true_tags=None, do_print=False):
        # print(f"Infering with {self.device}, model device: {self.model.device}")
        self.model.eval()
        # tokenize sentence
        # nlp = Indonesian()
        # nlp = spacy.load("en_core_web_sm")
        # tokens = [token.text for token in nlp(sentence)]
        sentences_tokens = [sent.split(' ') for sent in sentences]
        sent_len = [len(tokens) for tokens in sentences_tokens]
        max_sent_len = max(sent_len)
        max_word_len = max([len(token) for tokens in sentences_tokens for token in tokens])
        # transform to indices based on corpus vocab
        char_pad_idx, word_pad_idx = self.data.char_pad_idx, self.data.word_pad_idx
        numericalized_tokens = [[self.data.word_field.vocab.stoi[token.lower()] for token in tokens] + [word_pad_idx] * (max_sent_len - len(tokens))
                                for tokens in sentences_tokens]
        pad_chars = [self.data.char_field.vocab.stoi[char] for char in self.data.word_field.pad_token] + [char_pad_idx] * (max_word_len - len(self.data.word_field.pad_token))
        numericalized_chars = [[[self.data.char_field.vocab.stoi[char] for char in token] + [char_pad_idx] * (max_word_len - len(token))
                                for token in tokens] + [pad_chars] * (max_sent_len - len(tokens)) for tokens in sentences_tokens]

        # begin prediction
        token_tensor = torch.as_tensor(numericalized_tokens).transpose(0, 1).to(self.device)
        char_tensor = torch.as_tensor(numericalized_chars).to(self.device)
        predictions, _ = self.model(token_tensor, char_tensor)

        # convert results to tags
        predicted_tags = [[self.data.tag_field.vocab.itos[t] for t in preds[:length]] for preds, length in zip(predictions, sent_len)]

        return sentences_tokens, predicted_tags

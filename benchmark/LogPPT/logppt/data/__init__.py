import torch
from transformers import DataCollatorForTokenClassification

from logppt.data.load_data import *


class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        ori_labels = [feature['ori_labels'] for feature in features] if 'ori_labels' in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            # max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            batch['ori_labels'] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                                   ori_labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            batch["ori_labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                                   ori_labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

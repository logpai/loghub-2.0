from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    MODEL_MAPPING
)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
LR_SCHEDULER_TYPE = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]


@dataclass
class MainArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    mode: Optional[str] = field(
        default="prompt-tuning",
        metadata={"help": "Training model. choose from prompt-tuning or fine-tuning",
                  "choices": ["prompt-tuning", "fine-tuning"]}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A json file containing the training data."}
    )

    unlabeled_file: Optional[str] = field(
        default=None,
        metadata={"help": "A json file containing the unlabeled training data."}
    )

    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A json file containing the validation data."}
    )
    dev_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."}
    )
    text_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    max_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, "
                    "sequences shorter will be padded if `--pad_to_max_length` is passed."}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={"help": "If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used."}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the final model."}
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "A seed for reproducible training."}
    )

    shot: Optional[int] = field(
        default=50,
        metadata={"help": "The number of examples use for tuning"}
    )


@dataclass
class TrainArguments:
    per_device_train_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "Batch size (per device) for the training dataloader."}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "Batch size (per device) for the evaluation dataloader."}
    )
    learning_rate: Optional[float] = field(
        default=5e-5,
        metadata={"help": "Initial learning rate (after the potential warmup period) to use."}
    )
    weight_decay: Optional[float] = field(
        default=0.0,
        metadata={"help": "Weight decay to use."}
    )
    num_train_epochs: Optional[int] = field(
        default=20,
        metadata={"help": "Total number of training epochs to perform."}
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use.",
                  "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                              "constant_with_warmup"]}
    )
    num_warmup_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of steps for the warmup in the lr scheduler."}
    )

    wo_label_words: bool = field(
        default=False,
        metadata={"help": "If passed, don't select label words."}
    )

    no_label_words: Optional[int] = field(
        default=8,
        metadata={"help": "number of label words"}
    )

    def __post_init__(self):
        if self.lr_scheduler_type not in LR_SCHEDULER_TYPE:
            raise ValueError("Unknown learning rate scheduler, you should pick one in " + ",".join(LR_SCHEDULER_TYPE))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )


@dataclass
class TaskArguments:
    task_name: Optional[str] = field(
        default="log-parsing",
        metadata={"help": "The name of the task", "choices": ["log-parsing", "anomaly-detection"]}
    )

    dataset_name: Optional[str] = field(
        default="Apache",
        metadata={"help": "The name of the dataset"}
    )

    log_file: Optional[str] = field(
        default="Apache_2k.log",
        metadata={"help": "The name of the log file"}
    )

    task_output_dir: Optional[str] = field(
        default="outputs",
        metadata={"help": "The output directory of the log analytic task"}
    )

from transformers import AutoModelForSequenceClassification, AutoTokenizer, \
    Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from core.config import CONFIG
import torch
import torch.nn as nn
from datasets import Dataset, load_metric
import logging
import numpy as np

logger = logging.getLogger(__name__)

DEBERTA_V3_MODEL_NAME = "microsoft/deberta-v3-small"


def DebertaV3(num_classes: int):
    return AutoModelForSequenceClassification.from_pretrained(
        DEBERTA_V3_MODEL_NAME,
        num_labels=num_classes
    )


def Tokenizer():
    return AutoTokenizer.from_pretrained(DEBERTA_V3_MODEL_NAME)


class TextTrainerDebertaV3:
    _epochs_print = 20
    _num_epochs = 10
    _max_len = 512
    _output_dir = "hf_results"
    _tokenizer = None
    _trainer = None

    @classmethod
    def train(
        cls,
        train_ds: Dataset,
    ):
        logger.info("Initializing the text configurations...")
        classes = CONFIG.dataset_emotions()
        cls._tokenizer = Tokenizer()
        model = DebertaV3(num_classes=len(classes))
        tokenized_ds = train_ds.map(cls._tokenize)

        cls._trainer = Trainer(
            model=model,
            args=TrainingArguments(
                num_train_epochs=cls._num_epochs,
                save_strategy='epoch',
                logging_strategy='epoch',
                per_device_train_batch_size=4,
                report_to="none",
                do_eval=False,
                output_dir=cls._output_dir
            ),
            train_dataset=tokenized_ds,
            tokenizer=cls._tokenizer,
            compute_metrics=cls.compute_metrics
        )

        logger.info("Training..")
        cls._trainer.train()

        return model, cls._tokenizer

    @classmethod
    def eval(cls, test_ds: Dataset):
        tokenized_eval = test_ds.map(cls._tokenize)
        metrics = cls._trainer.evaluate(tokenized_eval)
        logger.info(metrics)

    @classmethod
    def _tokenize(cls, batch):
        return cls._tokenizer(
            batch["text"], max_length=cls._max_len, padding="longest", truncation=True
        )
        
    @classmethod
    def compute_metrics(cls, eval_pred):
        accuracy_metric = load_metric("accuracy")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

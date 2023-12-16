import os
import torch
import logging
from core.config import CONFIG
from scripts.get_dataloaders import get_dataloaders
from text.bert import Bert, BertTokenizer
from text.deberta import DebertaV3, DebertaV3Tokenizer
from text.train import TextTrainer

logger = logging.getLogger(__name__)


def only_text(model_class, tokenizer_class, save_path: str):
    logger.info("Getting the dataloaders..")
    train_dataloader, test_dataloader = get_dataloaders()
    logger.info("Starting the training process..")
    num_classes = len(CONFIG.dataset_emotions())
    model = model_class(num_classes)
    tokenizer = tokenizer_class()
    trainer = TextTrainer(model, tokenizer)
    trainer.train(train_dataloader)
    logger.info("Evaluating..")
    trainer.eval(test_dataloader)
    logger.info("Saving the model..")
    torch.save(model, os.path.join(CONFIG.saved_models_location(), save_path))


def only_text_using_bert():
    only_text(Bert, BertTokenizer, "bert_model.pt")


def only_text_using_deberta():
    only_text(DebertaV3, DebertaV3Tokenizer, "debertaV3_model.pt")

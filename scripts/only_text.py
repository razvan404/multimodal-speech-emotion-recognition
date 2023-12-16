import os
import torch
import logging
from core.config import CONFIG
from scripts.get_dataloaders import get_dataloaders
from text.bert import Bert
from text.deberta import DebertaV3
from text.train import TextTrainer

logger = logging.getLogger(__name__)


def only_text(model_class, save_path: str):
    logger.info("Getting the dataloaders..")
    train_dataloader, test_dataloader = get_dataloaders()
    logger.info("Starting the training process..")
    num_classes = len(CONFIG.dataset_emotions())
    model = model_class(num_classes)
    trainer = TextTrainer(model)
    trainer.train(train_dataloader)
    torch.save(model, os.path.join(CONFIG.saved_models_location(), save_path))
    logger.info("Evaluating..")
    trainer.eval(test_dataloader)
    logger.info("Saving the model..")


def only_text_using_bert():
    only_text(Bert, "bert_model.pt")


def only_text_using_deberta():
    only_text(DebertaV3, "deberta_model.pt")

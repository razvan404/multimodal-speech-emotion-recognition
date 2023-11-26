import os
import torch
import logging
from core.config import CONFIG
from scripts.get_dataloaders import get_dataloaders
from scripts.get_hf_datasets import get_hf_datasets
from text.train import TextTrainer
from text.deberta import TextTrainerDebertaV3

logger = logging.getLogger(__name__)


def only_text():
    logger.info("Getting the dataloaders..")
    train_dataloader, test_dataloader = get_dataloaders()
    logger.info("Starting the training process..")
    model, tokenizer = TextTrainer.train(train_dataloader)
    logger.info("Evaluating..")
    TextTrainer.eval(model, tokenizer, test_dataloader)
    logger.info("Saving the model..")
    torch.save(model, os.path.join(
        CONFIG.saved_models_location(), "text_model.pt"))


def deberta():
    logger.info("Getting the dataloaders..")
    train_ds, test_ds = get_hf_datasets()
    logger.info("Starting the training process..")
    model = TextTrainerDebertaV3.train(train_ds.get_hf_dataset())
    logger.info("Evaluating..")
    TextTrainerDebertaV3.eval(test_ds.get_hf_dataset())
    logger.info("Saving the model..")
    torch.save(model, os.path.join(
        CONFIG.saved_models_location(), "deberta_model.pt"))

import os
import torch
import logging
from core.config import CONFIG
from scripts.get_dataloaders import get_dataloader
from text.train import TextTrainer

logger = logging.getLogger(__name__)


def train_text(model_class, save_path: str):
    logger.info("Getting the dataloader...")
    train_dataloader = get_dataloader("train", True)
    logger.info("Starting the training process...")
    num_classes = len(CONFIG.dataset_emotions())
    model = model_class(num_classes)
    print(model)
    input()
    trainer = TextTrainer(model)
    trainer.train(train_dataloader)
    torch.save(model, os.path.join(CONFIG.saved_models_location(), save_path))
    logger.info("Saving the model...")


def eval_text(load_path: str):
    logger.info("Getting the model and the dataloader...")
    emotions = CONFIG.dataset_emotions()
    model = torch.load(os.path.join(CONFIG.saved_models_location(), load_path))
    test_dataloader = get_dataloader("test", False)
    logger.info("Evaluating the model...")
    trainer = TextTrainer(model)
    trainer.eval(test_dataloader, emotions)

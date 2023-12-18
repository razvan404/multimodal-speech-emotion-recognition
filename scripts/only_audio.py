import logging
import os

import torch

from audio.train import AudioTrainer
from core.config import CONFIG
from scripts.get_dataloaders import get_dataloader

logger = logging.getLogger(__name__)


def train_audio(model_class, save_path: str):
    logger.info("Getting the dataloaders...")
    train_dataloader, test_dataloader = get_dataloader()
    logger.info("Starting the training process...")
    num_classes = len(CONFIG.dataset_emotions())
    model = model_class(num_classes)
    AudioTrainer.train(model, train_dataloader, num_classes)
    AudioTrainer.eval(model, test_dataloader)
    torch.save(model, os.path.join(CONFIG.saved_models_location(), "audio_model.pt"))

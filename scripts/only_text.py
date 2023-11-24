import os

import torch

from core.config import CONFIG
from scripts.get_dataloaders import get_dataloaders
from text.train import TextTrainer


def only_text():
    train_dataloader, test_dataloader = get_dataloaders()
    model, tokenizer = TextTrainer.train(train_dataloader)
    torch.save(model, os.path.join(CONFIG.saved_models_location(), "text_model.pt"))
    TextTrainer.eval(model, tokenizer, test_dataloader)

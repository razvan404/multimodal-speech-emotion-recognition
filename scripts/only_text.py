import os

import torch

from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader
from text.train_text import TextTrainer


def only_text():
    train_dataloader = IemocapDataLoader(
        CONFIG.dataset_path(),
        CONFIG.dataset_preprocessed_file_path(),
        CONFIG.dataset_emotions(),
        "train",
        **CONFIG.dataloader_dict(),
    )
    test_dataloader = IemocapDataLoader(
        CONFIG.dataset_path(),
        CONFIG.dataset_preprocessed_file_path(),
        CONFIG.dataset_emotions(),
        "test",
        **CONFIG.dataloader_dict(),
    )
    model, tokenizer = TextTrainer.train(train_dataloader)
    TextTrainer.eval(model, tokenizer, test_dataloader)
    torch.save(model, os.path.join(CONFIG.saved_models_location(), "text_model.pt"))

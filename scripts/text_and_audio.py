import os

import torch

from core.config import CONFIG
from fusion.model import CombinedAudioTextModel
from fusion.train import FusionTrainer
from scripts.get_dataloaders import get_dataloader


def text_and_audio():
    train_dataloader, test_dataloader = get_dataloader()
    classes = CONFIG.dataset_emotions()
    model = CombinedAudioTextModel(len(classes))
    _ = FusionTrainer.train(model, train_dataloader, len(classes))
    torch.save(
        model, os.path.join(CONFIG.saved_models_location(), "multimodal_model.pt")
    )

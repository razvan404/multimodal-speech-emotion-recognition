import os

import torch

from audio.alexnet import alexnet, modified_alexnet
from audio.train import AudioTrainer
from core.config import CONFIG
from scripts.get_dataloaders import get_dataloaders


def get_model():
    emotions = CONFIG.dataset_emotions()
    original_model = alexnet(CONFIG.pretrained_alexnet_url(), pretrained=True)
    original_dict = original_model.state_dict()
    modified_model = modified_alexnet(pretrained=False, num_classes=len(emotions))
    modified_model_dict = modified_model.state_dict()
    pretrained_modified_model_dict = {
        key: value for key, value in original_dict.items() if key in modified_model_dict
    }
    modified_model_dict.update(pretrained_modified_model_dict)
    modified_model.load_state_dict(modified_model_dict)
    return modified_model


def only_audio():
    emotions = CONFIG.dataset_emotions()
    model = get_model()
    train_dataloader, test_dataloader = get_dataloaders()
    AudioTrainer.train(model, train_dataloader, len(emotions))
    AudioTrainer.eval(model, test_dataloader)
    torch.save(model, os.path.join(CONFIG.saved_models_location(), "audio_model.pt"))

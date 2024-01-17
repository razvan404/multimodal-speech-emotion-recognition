import logging
import os

import torch

from audio.trainer import AudioTrainer
from audio.wav2vec2 import Wav2Vec2
from core.config import CONFIG
from core.trainer import AbstractTrainer
from scripts.get_dataloaders import get_dataloader
from text.deberta import DebertaV3
from text.trainer import TextTrainer

logger = logging.getLogger(__name__)


class TrainerOps:
    @classmethod
    def create_or_load_text_trainer(
        cls, load_path: str | None = None, load_state_dict: bool = False
    ):
        num_classes = len(CONFIG.dataset_emotions())
        if load_path is not None:
            if load_state_dict:
                model = DebertaV3(num_classes)
                state_dict = torch.load(
                    os.path.join(CONFIG.saved_models_location(), load_path)
                )
                model.load_state_dict(state_dict)
            else:
                model = torch.load(
                    os.path.join(CONFIG.saved_models_location(), load_path)
                )
        else:
            model = DebertaV3(num_classes)
        return TextTrainer(model)

    @classmethod
    def create_or_load_audio_trainer(
        cls, load_path: str | None = None, load_state_dict: bool = False
    ):
        num_classes = len(CONFIG.dataset_emotions())
        if load_path is not None:
            if load_state_dict:
                model = Wav2Vec2(num_classes)
                state_dict = torch.load(
                    os.path.join(CONFIG.saved_models_location(), load_path)
                )
                model.load_state_dict(state_dict)
            else:
                model = torch.load(
                    os.path.join(CONFIG.saved_models_location(), load_path)
                )
        else:
            model = Wav2Vec2(num_classes)
        return AudioTrainer(model)

    @classmethod
    def train(cls, trainer: AbstractTrainer):
        logger.info("Getting the train dataloader...")
        train_dataloader = get_dataloader("train", shuffle=True)
        logger.info("Starting the training process...")
        trainer.train(train_dataloader)

    @classmethod
    def save(
        cls, trainer: AbstractTrainer, save_path: str, save_state_dict: bool = False
    ):
        logger.info("Saving the model...")
        if save_state_dict:
            torch.save(
                trainer.model.state_dict(),
                os.path.join(CONFIG.saved_models_location(), save_path),
            )
        else:
            torch.save(
                trainer.model, os.path.join(CONFIG.saved_models_location(), save_path)
            )

    @classmethod
    def evaluate(cls, trainer: AbstractTrainer):
        logger.info("Getting the eval dataloader...")
        emotions = CONFIG.dataset_emotions()
        test_dataloader = get_dataloader("test", False)
        logger.info("Evaluating the model...")
        trainer.eval(test_dataloader, emotions)

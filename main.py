import os
import random
import numpy as np
import torch

from audio.extractor import MfccExtractor
from audio.timnet import TimNetClassifier
from core.config import CONFIG
from scripts.only_audio import train_audio
from scripts.only_text import train_text, eval_text
from scripts.preprocess_data import (
    process_audio_data_to_pickle,
    process_raw_data_to_pickle,
    process_text_data_to_pickle,
)
from scripts.text_and_audio import text_and_audio
import logging

from text.deberta import DebertaV3Tokenizer, DebertaV3

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def prepare_env():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    CONFIG.load_config("config.yaml")


if __name__ == "__main__":
    # 0. Prepare the environment
    prepare_env()
    # 1. Prepare audio paths and raw text
    # process_raw_data_to_pickle("audio_and_text.pkl")
    # 2. Turn the raw audio file names into mfccs
    # process_audio_data_to_pickle(
    #     "audio_and_text.pkl", "mfccs_and_text.pkl", MfccExtractor()
    # )
    # 3. Turn the raw text file into tokens
    # process_text_data_to_pickle(
    #     "mfccs_and_text.pkl", "mfccs_and_tokens.pkl", DebertaV3Tokenizer()
    # )
    # 4. Train the only text model
    # train_text(DebertaV3, "deberta_model.pt")
    # 4.1. Evaluate the text model
    eval_text("deberta_model.pt")
    # 5. Train the only audio model
    # train_audio(TimNetClassifier, "timnet_model.pt")
    # 6. Train the fusion model
    # text_and_audio()

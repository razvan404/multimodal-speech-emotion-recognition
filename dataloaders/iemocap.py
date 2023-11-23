import os

import numpy as np
import pandas as pd
import torch.nn.functional
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

from audio.extractor import MfccExtractor, Spectrogram3DExtractor


class IemocapDataset(Dataset):
    def __init__(self, dataset_path: str, audio_and_text_csv: str, emotions: str):
        self._dataset_path = dataset_path
        self._emotions = emotions
        self._dataframe = pd.read_csv(audio_and_text_csv)
        self._dataframe = self._dataframe[self._dataframe["emotion"].isin(emotions)]

    def __getitem__(self, index: int):
        audio, text, emotion = self._dataframe.iloc[index]
        emotion_one_hot = torch.tensor(
            np.array(self._emotions == emotion, dtype=np.int8)
        )
        session_id = int(audio[3:5])
        wav_path = os.path.join(
            self._dataset_path,
            f"Session{session_id}",
            "sentences",
            "wav",
            audio[:-5],
            f"{audio}.wav",
        )
        audio_spec = Spectrogram3DExtractor.extract(wav_path)
        return audio_spec, text, emotion_one_hot

    def __len__(self):
        return len(self._dataframe)


def IemocapDataLoader(
    dataset_path: str, audio_and_text_csv: str, emotions: list[str], **kwargs
):
    return DataLoader(
        IemocapDataset(dataset_path, audio_and_text_csv, emotions), **kwargs
    )

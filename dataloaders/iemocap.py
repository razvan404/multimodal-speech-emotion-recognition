import os

import numpy as np
import pandas as pd
import torch.nn.functional

from torch.utils.data import Dataset, DataLoader

from audio.extractor import Spectrogram3DExtractor


class IemocapDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        audio_and_text_csv: str,
        emotions: list[str],
        division: str,
    ):
        self._dataset_path = dataset_path
        self._emotions = np.array(emotions)
        self._dataframe = pd.read_csv(audio_and_text_csv)
        self._dataframe = self._dataframe[self._dataframe["emotion"].isin(emotions)][
            0::20
        ]
        rows_80_percent = int(0.8 * len(self._dataframe))
        if division == "train":
            self._dataframe = self._dataframe.iloc[:rows_80_percent, :]
        elif division == "test":
            self._dataframe = self._dataframe.iloc[rows_80_percent:, :]
        else:
            raise ValueError("Invalid dataset divison")
        self._extractor = Spectrogram3DExtractor

    def __getitem__(self, index: int):
        audio, text, emotion = self._dataframe.iloc[index]
        emotion_index = torch.tensor(np.where(self._emotions == emotion)[0])
        session_id = int(audio[3:5])
        wav_path = os.path.join(
            self._dataset_path,
            f"Session{session_id}",
            "sentences",
            "wav",
            audio[:-5],
            f"{audio}.wav",
        )
        audio_spec = self._extractor.extract(wav_path)
        return audio_spec, text, emotion_index

    def __len__(self):
        return len(self._dataframe)


def IemocapDataLoader(
    dataset_path: str,
    audio_and_text_csv: str,
    emotions: list[str],
    division: str,
    **kwargs,
):
    return DataLoader(
        IemocapDataset(dataset_path, audio_and_text_csv, emotions, division), **kwargs
    )

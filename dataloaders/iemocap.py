import os
import logging

import numpy as np
import pandas as pd
import torch.nn.functional

from torch.utils.data import Dataset, DataLoader

from audio.extractor import Spectrogram3DExtractor

logger = logging.getLogger(__name__)


class IemocapDataset(Dataset):
    def _extract_data_from_audio(self, audio_path: str):
        session_id = int(audio_path[3:5])
        wav_path = os.path.join(
            self._dataset_path,
            f"Session{session_id}",
            "sentences",
            "wav",
            audio_path[:-5],
            f"{audio_path}.wav",
        )
        return self._audio_extractor.extract(wav_path)

    def __init__(
        self,
        dataset_path: str,
        audio_and_text_csv: str,
        emotions: list[str],
        split: str,
    ):
        self._dataset_path = dataset_path
        self._emotions = np.array(emotions)
        self._dataframe = pd.read_csv(audio_and_text_csv)
        self._audio_extractor = Spectrogram3DExtractor

        self._dataframe = self._dataframe.loc[
            (self._dataframe["emotion"].isin(emotions))
            & (
                self._dataframe["audio"].apply(
                    lambda path: self._extract_data_from_audio(path).shape[2] > 65
                )
            )
        ]
        rows_80_percent = int(0.8 * len(self._dataframe))
        if split == "train":
            self._dataframe = self._dataframe.iloc[:rows_80_percent, :]
        elif split == "test":
            self._dataframe = self._dataframe.iloc[rows_80_percent:, :]
        else:
            raise ValueError("Invalid dataset split")
        logger.info(f"Loaded {split} dataset. Size: {len(self)}")
        emotions_str = ""
        for emotion in emotions:
            emotions_str += f"{emotion} - {self._dataframe[self._dataframe['emotion'] == emotion]['emotion'].count()} | "
        emotions_str = emotions_str[:-3]
        logger.info(f"Each emotion percentages: {emotions_str}")

    def __getitem__(self, index: int):
        audio, text, emotion = self._dataframe.iloc[index]
        emotion_index = torch.tensor(np.where(self._emotions == emotion)[0])
        audio_spec = self._extract_data_from_audio(audio)
        return audio_spec, text, emotion_index

    def __len__(self):
        return len(self._dataframe)


def IemocapDataLoader(
    dataset_path: str,
    audio_and_text_csv: str,
    emotions: list[str],
    split: str,
    **kwargs,
):
    return DataLoader(
        IemocapDataset(dataset_path, audio_and_text_csv, emotions, split), **kwargs
    )

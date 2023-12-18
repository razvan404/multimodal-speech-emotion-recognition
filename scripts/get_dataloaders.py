import os.path

import pandas as pd

from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader


def get_dataloader(
    split: str | list[str] | None = None, shuffle: bool | list[bool] | None = None
):
    if split is None:
        split = ["train", "test"]
    if shuffle is None:
        shuffle = [True, False]
    dataframe = pd.read_pickle(
        os.path.join(CONFIG.dataset_preprocessed_dir_path(), "mfccs_and_tokens.pkl")
    )
    if isinstance(split, str):
        return IemocapDataLoader(
            CONFIG.dataset_path(),
            dataframe,
            CONFIG.dataset_emotions(),
            split,
            **CONFIG.dataloader_dict(),
            shuffle=shuffle,
        )
    else:
        return [
            IemocapDataLoader(
                CONFIG.dataset_path(),
                dataframe,
                CONFIG.dataset_emotions(),
                split_item,
                **CONFIG.dataloader_dict(),
                shuffle=shuffle_item,
            )
            for split_item, shuffle_item in zip(split, shuffle)
        ]

import os.path

import pandas as pd

from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader


def get_dataloaders():
    dataframe = pd.read_pickle(
        os.path.join(CONFIG.dataset_preprocessed_dir_path(), "mfccs_and_tokens.pkl")
    )
    train_dataloader = IemocapDataLoader(
        CONFIG.dataset_path(),
        dataframe,
        CONFIG.dataset_emotions(),
        "train",
        **CONFIG.dataloader_dict(),
        shuffle=True,
    )
    test_dataloader = IemocapDataLoader(
        CONFIG.dataset_path(),
        dataframe,
        CONFIG.dataset_emotions(),
        "test",
        **CONFIG.dataloader_dict(),
        shuffle=False,
    )
    return train_dataloader, test_dataloader

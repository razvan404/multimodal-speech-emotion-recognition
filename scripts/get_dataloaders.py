from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader


def get_dataloaders():
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
    return train_dataloader, test_dataloader

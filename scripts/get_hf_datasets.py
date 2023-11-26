from core.config import CONFIG
from dataloaders.iemocap import IemocapHFDataset


def get_hf_datasets():
    train_dataset = IemocapHFDataset(
        CONFIG.dataset_path(),
        CONFIG.dataset_preprocessed_file_path(),
        CONFIG.dataset_emotions(),
        "train",
    )
    test_dataset = IemocapHFDataset(
        CONFIG.dataset_path(),
        CONFIG.dataset_preprocessed_file_path(),
        CONFIG.dataset_emotions(),
        "test",
    )
    return train_dataset, test_dataset

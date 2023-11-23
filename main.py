from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader

if __name__ == "__main__":
    CONFIG.load_config("config.yaml")
    dataloader = IemocapDataLoader(
        CONFIG.dataset_path(),
        CONFIG.dataset_preprocessed_file_path(),
        CONFIG.dataset_emotions(),
        **CONFIG.dataloader_dict(),
    )
    print(len(dataloader), dataloader.batch_size)

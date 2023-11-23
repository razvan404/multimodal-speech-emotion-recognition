from core.config import CONFIG
from dataloaders.iemocap import IemocapDataLoader

if __name__ == "__main__":
    CONFIG.load_config("config.yaml")
    dataloader = IemocapDataLoader(CONFIG.dataset_path(), **CONFIG.dataloader_dict())
    print(dataloader.batch_size)

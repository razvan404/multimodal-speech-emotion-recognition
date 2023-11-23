from torch.utils.data import Dataset, DataLoader


class IemocapDataset(Dataset):
    def __init__(self, filename: str):
        pass

    def __getitem__(self, index: int):
        pass

    def __len__(self):
        return 1


def IemocapDataLoader(filename: str, **kwargs):
    return DataLoader(IemocapDataset(filename), **kwargs)

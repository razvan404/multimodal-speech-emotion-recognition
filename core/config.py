import os
import yaml


class CONFIG:
    _dict: dict

    @classmethod
    def load_config(cls, filename: str):
        with open(filename, "r") as file:
            cls._dict = yaml.safe_load(file)

    @classmethod
    def _path_from_data(cls, path: str):
        return os.path.join(cls._dict["data_source"]["path"], path)

    @classmethod
    def dataset_path(cls):
        return cls._path_from_data(cls._dict["dataset"]["name"])

    @classmethod
    def dataloader_dict(cls):
        return cls._dict["dataloader"]

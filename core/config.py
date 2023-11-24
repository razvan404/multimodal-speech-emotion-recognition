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

    @classmethod
    def dataset_preprocessed_dir_path(cls):
        return os.path.join(
            cls.dataset_path(),
            cls._dict["dataset_specific"][cls._dict["dataset"]["name"]]["preprocessed"][
                "dir"
            ],
        )

    @classmethod
    def dataset_preprocessed_file_path(cls):
        return os.path.join(
            cls.dataset_preprocessed_dir_path(),
            cls._dict["dataset_specific"][cls._dict["dataset"]["name"]]["preprocessed"][
                "file"
            ],
        )

    @classmethod
    def dataset_emotions(cls):
        return cls._dict["dataset_specific"][cls._dict["dataset"]["name"]]["emotions"]

    @classmethod
    def saved_models_location(cls):
        return cls._dict["models"]["save_location"]

    @classmethod
    def pretrained_alexnet_url(cls):
        return cls._dict["models"]["pretrained_alexnet"]

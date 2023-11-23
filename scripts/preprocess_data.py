import os

from core.config import CONFIG
from preprocessing.iemocap import IemocapPreprocessor


def preprocess_data():
    preprocessor = IemocapPreprocessor(CONFIG.dataset_path())
    df = preprocessor.generate_dataframe()

    preprocessed_path = CONFIG.dataset_preprocessed_dir_path()
    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path)

    df.to_csv(CONFIG.dataset_preprocessed_file_path(), index=False)

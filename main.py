from core.config import CONFIG
from scripts.preprocess_data import preprocess_data

if __name__ == "__main__":
    CONFIG.load_config("config.yaml")
    preprocess_data()

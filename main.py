from core.config import CONFIG
from scripts.only_text import only_text

if __name__ == "__main__":
    CONFIG.load_config("config.yaml")
    only_text()

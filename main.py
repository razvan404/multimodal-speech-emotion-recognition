from core.config import CONFIG
from scripts.only_audio import only_audio
from scripts.only_text import only_text_using_bert, only_text_using_deberta
from scripts.text_and_audio import text_and_audio
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    CONFIG.load_config("config.yaml")
    only_text_using_deberta()
    # only_audio()
    # text_and_audio()

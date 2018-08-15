import sys
import logging

from .ape import Ape
from .config import ONTOLOGY_PATH, EMBEDDING_PATH, LOG_LEVEL

logging.basicConfig(
    level=LOG_LEVEL,
    stream=sys.stdout,
)

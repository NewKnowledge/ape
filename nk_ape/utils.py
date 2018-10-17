import os
import time
from datetime import datetime

import numpy as np
from inflection import underscore
from nk_logger import get_logger

logger = get_logger(__name__)


DASHES_TO_SPACES = {"_": " ", "-": " "}
REMOVE_PAREN = {"(": "", ")": ""}


def normalize_text(
    text,
    to_list=True,
    replace_chars={"_": " ", "-": " ", "(": "", ")": "", ".": "", ",": "", "!": "", "?": ""},
):
    text = underscore(text)  # converts to snake_case
    for old, new in replace_chars.items():
        text = text.replace(old, new)
    if to_list:
        return text.split(" ")
    else:
        return text


def unit_norm_rows(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def mean_of_rows(vectors):
    return np.mean(vectors, axis=0)


def max_of_rows(vectors):
    return np.max(vectors, axis=0)


def in_vocab(word_list, model):
    if isinstance(word_list, str):
        word_list = word_list.split(" ")
    return all([word in model.vocab for word in word_list])


def get_dropped(old, new):
    return set(old).difference(set(new))


def no_op():
    return None


def path_to_name(path):
    return os.path.basename(path).split(".")[0]


def timeit(func, args=None):
    start = time.time()
    logger.info(f"calling {func.__name__} \n")
    result = func(*args) if args else func()
    logger.info(f"{func.__name__} took {time.time() - start} seconds \n\n")
    return result


def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")

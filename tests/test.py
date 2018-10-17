""" Tests for ape """
import logging

from nk_logger import get_logger

from nk_ape import Ape

logger = get_logger(f"ape.{__name__}", level=logging.INFO)

ape = Ape()


def test_top_classes(n_classes=5):

    logger.debug("starting test")
    input_words = ["gorilla", "chimp", "orangutan", "gibbon", "human"]
    result = ape.get_top_classes(input_words, n_classes=n_classes)
    assert isinstance(result, list)
    assert len(result) == n_classes
    res = result[0]
    assert isinstance(res, dict)
    assert "class" in res.keys() and isinstance(res["class"], str)
    assert "score" in res.keys() and isinstance(res["score"], float)
    logger.debug("top classes test complete")


def test_empty(n_classes=5):

    result = ape.get_top_classes([], n_classes=n_classes)
    logger.info(result)

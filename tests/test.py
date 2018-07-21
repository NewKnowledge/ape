''' Tests for ape '''
from nk_ape import Ape
import logging

ape = Ape(verbose=True)


def test_top_classes(n_classes=5):

    logging.debug('starting test')
    input_words = ['gorilla', 'chimp', 'orangutan', 'gibbon', 'human']
    result = ape.get_top_classes(input_words, n_classes=n_classes)
    assert isinstance(result, list)
    assert len(result) == n_classes
    res = result[0]
    assert isinstance(res, dict)
    assert 'class' in res.keys() and isinstance(res['class'], str)
    assert 'score' in res.keys() and isinstance(res['score'], float)
    logging.debug('top classes test complete')

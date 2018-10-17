import logging

import numpy as np

from gensim.models import KeyedVectors, Word2Vec

from .config import EMBEDDING_PATH
from .utils import mean_of_rows, no_op

from nk_logger import get_logger

logger = get_logger(__name__)


class Embedding:
    """ Load a word2vec embedding from a file """

    def __init__(self, embedding_path=EMBEDDING_PATH, embed_agg_func=mean_of_rows):

        self.embed_agg_func = embed_agg_func

        logger.info("loading word2vec embedding model")
        try:
            binary = ".bin" in embedding_path
            model = KeyedVectors.load_word2vec_format(embedding_path, binary=binary)
        except UnicodeDecodeError as err:
            logger.warning(f"error loading model: {err}")
            logger.info("trying different load function")
            model = KeyedVectors.load(embedding_path)
        # we only use the embedding vectors (no training), so we can get rid of the rest of the model
        self.model = model.wv
        del model
        logger.info("succeeded in loading embedding model")

    def remove_out_of_vocab(self, word_groups):
        if isinstance(word_groups, str):
            word_groups = word_groups.split(" ")

        if not isinstance(word_groups, np.ndarray):
            word_groups = np.array(word_groups)

        # removes all word lists with any oov words
        in_vocab = [self.in_vocab(group) for group in word_groups]
        logger.info(
            "dropping {0} out of {1} values for having out-of-vocab words".format(
                len(word_groups) - sum(in_vocab), len(word_groups)
            )
        )
        return word_groups[in_vocab]

    def embed_word(self, word):
        return self.model[word]

    def embed_multi_words(self, word_list):
        return self.embed_agg_func([self.model[word] for word in word_list])

    def n_similarity(self, words, classes):
        return self.model.n_similarity(words, classes)

    def in_vocab(self, word_list):
        if isinstance(word_list, str):
            word_list = word_list.split(" ")
        return all([word in self.model.vocab for word in word_list])

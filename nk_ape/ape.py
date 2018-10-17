""" Core Ape client """

import numpy as np
from inflection import pluralize

from .class_tree import EmbeddedClassTree, tree_score
from .config import EMBEDDING_PATH, ONTOLOGY_PATH
from .embedding import Embedding
from .utils import mean_of_rows, normalize_text, unit_norm_rows
from nk_logger import get_logger

logger = get_logger(__name__)


class Ape:
    def __init__(
        self,
        embedding_path=EMBEDDING_PATH,
        ontology_path=ONTOLOGY_PATH,
        row_agg_func=mean_of_rows,
        tree_agg_func=np.mean,
    ):

        logger.info(
            f"initializing ape, loading word embedding from {embedding_path}, this can take a while..."
        )
        self.embedding = Embedding(embedding_path=embedding_path)
        self.tree = EmbeddedClassTree(self.embedding, tree_path=ontology_path)

        self.row_agg_func = row_agg_func
        self.tree_agg_func = tree_agg_func

        logger.info("ape initialized")

    @property
    def classes(self):
        return self.tree.classes

    def format_input(self, input_text):
        """' format into list of lists of single words, removing words outside the word embedding vocab """
        logger.debug("normalizing input text and removing out-of-vocab words")
        word_groups = np.array([normalize_text(text) for text in input_text])
        return self.embedding.remove_out_of_vocab(word_groups)

    def compute_similarity_matrix(self, input_vectors):
        """ compute cosine similarity bt embedded data and ontology classes """
        logger.debug("computing similarity matrix between class and input vectors")
        return np.dot(input_vectors, self.tree.class_vectors.T)

    def aggregate_tree_scores(self, scores):
        # convert score to dict that maps class to score if needed
        score_map = scores if isinstance(scores, dict) else dict(zip(self.classes, scores))

        # aggregate score over tree structure
        agg_score_map = tree_score(score_map, self.tree, self.tree_agg_func)

        # convert returned score map back to array, make float64 to be json-serializable (for some reason float32 is not)
        return np.array([agg_score_map[cl] for cl in self.classes], dtype=np.float64)

    def get_class_scores(self, input_text):

        # clean/format text, embed input to get input_vectors
        input_text = self.format_input(input_text)
        # TODO check for multi-word text here (then use embedding.embed_multi_words or embed_word)
        # input_vectors = np.array([self.embedding.embed_word(word) for word in input_text])
        input_vectors = np.array([self.embedding.embed_multi_words(words) for words in input_text])
        input_vectors = unit_norm_rows(input_vectors)

        sim_matrix = self.compute_similarity_matrix(input_vectors)

        logger.debug("aggregating row scores")
        sim_scores = self.row_agg_func(sim_matrix)

        logger.debug("aggregating tree scores")
        return self.aggregate_tree_scores(sim_scores)

    def get_top_classes(self, input_text, n_classes=10):
        """ takes a list of strings as input text and returns the top n classes """
        # TODO control max length of input text?
        # TODO set score threshold instead of always top n classes?
        if input_text:
            scores = self.get_class_scores(input_text)
            sort_inds = np.argsort(scores)[::-1][:n_classes]
            top_classes = self.classes[sort_inds]
            top_scores = scores[sort_inds]

            return [
                {"class": ont_class, "score": score}
                for ont_class, score in zip(top_classes, top_scores)
            ]
        logger.info("empty input text list")
        return []

    def get_description(self, input_text):
        scores = self.get_class_scores(input_text)
        top_word = self.classes[np.argmax(scores)]
        logger.info(f"\n\nselecting top word as description: {top_word}\n\n")
        return f"The given text can be summarized as {pluralize(top_word)}"

    def predict(self, input_text, n_classes=10):
        return self.get_top_classes(input_text, n_classes=n_classes)

import json

import numpy as np

import ontospy
from inflection import underscore

from .config import ONTOLOGY_PATH
from .embedding import Embedding
from .utils import DASHES_TO_SPACES, REMOVE_PAREN, no_op, path_to_name, unit_norm_rows

from nk_logger import get_logger

logger = get_logger(__name__)


class EmbeddedClassTree:
    def __init__(self, embedding_model, tree_path=ONTOLOGY_PATH, embed_classes=True):

        assert isinstance(embedding_model, Embedding)
        self.embedding = embedding_model

        logger.info("loading ontology tree")
        self.tree = self.load_tree(tree_path)
        self.classes = np.array(list(self.tree.keys()))

        if embed_classes:
            # TODO why would we not want to embed classes?
            self.embed_classes()

    def embed_classes(self, classes=None):
        classes = classes if classes else self.classes
        classes = [cl.split(" ") if isinstance(cl, str) else cl for cl in classes]
        self.class_vectors = np.array(
            [self.embedding.embed_multi_words(words) for words in classes]
        )
        self.class_vectors = unit_norm_rows(self.class_vectors)
        return self.class_vectors

    def normalize_class_tree(self, tree):
        """ filter out keys with out-of-vocab words -- all words in class
            name must be in vocab
        """
        tree = {name: rels for (name, rels) in tree.items() if self.embedding.in_vocab(name)}
        classes = list(tree.keys())  # filtered class list

        # remove filtered classes from parent and child lists
        for cl, rels in tree.items():
            tree[cl]["children"] = [child for child in rels["children"] if (child in classes)]
            tree[cl]["parents"] = [parent for parent in rels["parents"] if (parent in classes)]

        return tree

    def load_tree(self, tree_path="ontologies/class-tree_dbpedia_2016-10.json"):
        with open(tree_path, "r") as tree_file:
            tree = json.load(tree_file)
        return self.normalize_class_tree(tree)


#  ontology utility functions:


def get_leaves(tree):
    return {name: rels for (name, rels) in tree.items() if not rels.get("children")}


def tree_score(score_map, tree, agg_func):

    tree = tree.tree if isinstance(tree, EmbeddedClassTree) else tree

    agg_score = {}
    processed = set()

    def all_children_aggd(node):
        return all([agg_score.get(ch) for ch in tree[node]["children"]])

    def process_layer(layer):
        assert layer
        for node in layer:
            agg_score[node] = apply_agg_func(node, score_map, tree, agg_score, agg_func)
            # agg_score[node] = apply_agg_func(node_score, relations, agg_score, agg_func)
            processed.add(node)

    all_nodes = set(tree.keys())
    layer = get_leaves(tree).keys()
    process_layer(layer)

    # while there are still unprocessed nodes, move up heirarchy
    while all_nodes.difference(processed):
        # get parents of previous layers
        layer = set.union(*[set(tree[node]["parents"]) for node in layer])
        # remove already processed
        layer = layer.difference(processed)
        # keep nodes where all child values have been computed
        layer = [n for n in layer if all_children_aggd(n)]
        # aggregate score for selected parents
        process_layer(layer)

    return agg_score


def apply_agg_func(node_name, score_map, tree, agg_score, agg_func=np.max):
    score_list = [score_map[node_name]]
    children = tree[node_name].get("children")
    # if current node has children:
    if children:
        child_agg_scores = [agg_score.get(child) for child in children]
        # make sure all children have agg_scores already
        if all(child_agg_scores):
            # add their agg scores to list
            score_list = score_list + child_agg_scores

    return agg_func(score_list)


def to_class_name(onto_class_obj, replace_chars=[DASHES_TO_SPACES, REMOVE_PAREN]):
    # if list of dicts given, concat their values
    if not isinstance(replace_chars, dict):
        replace_chars = {
            key: val for replace_map in replace_chars for key, val in replace_map.items()
        }

    # make lowercase and replace characters
    name = underscore(str(onto_class_obj.bestLabel()))
    for old, new in replace_chars.items():
        name = name.replace(old, new)

    return name


def has_relations(class_relations):
    return (len(class_relations["children"]) > 0) or (len(class_relations["parents"]) > 0)


def get_tree_file_name(ontology_name, prune=True):
    return f'class-tree_{ontology_name}{"_pruned" if prune else ""}.json'


def generate_class_tree_file(ontology_path="ontologies/dbpedia_2016-10.nt", prune=False):
    logger.info(f"loading ontology: {ontology_path}")
    ont = ontospy.Ontospy(ontology_path)

    all_classes = set()
    logger.info("building class relationships")
    relationships = {}
    for cl in ont.classes:
        relationships[to_class_name(cl)] = {
            "parents": [to_class_name(p) for p in cl.parents()],
            "children": [to_class_name(c) for c in cl.children()],
        }

        parents = {to_class_name(p) for p in cl.parents()}
        children = {to_class_name(c) for c in cl.children()}
        all_classes = all_classes.union(parents, children, set([to_class_name(cl)]))

    logger.info(f"pre prune relationships length: {len(relationships.keys())}")

    if prune:
        # remove all isolated classes (dont have children or parents)
        relationships = {
            name: rels for (name, rels) in relationships.items() if has_relations(rels)
        }

    logger.info(f"number of ontology classes: {len(ont.classes)}")

    logger.info(f"number of all classes (including children & parents): {len(all_classes)}")

    logger.info(f"number of relationships keys: {len(relationships.keys())}")

    logger.info(f"classes minus rel keys: {all_classes.difference(set(relationships.keys()))}")

    logger.info("writing class relationships to file")
    ontology_name = path_to_name(ontology_path)
    file_name = f"ontologies/class-tree_{ontology_name}{'_pruned' if prune else ''}.json"
    with open(file_name, "w") as rels_file:
        json.dump(relationships, rels_file, indent=4)


if __name__ == "__main__":
    generate_class_tree_file()

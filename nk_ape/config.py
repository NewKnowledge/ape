import os
import logging

EMBEDDING_PATH = os.getenv('EMBEDDING_PATH', 'nk_ape/embeddings/wiki2vec/en.model')
ONTOLOGY_PATH = os.getenv('ONTOLOGY_PATH', 'nk_ape/ontologies/class-tree_dbpedia_2016-10.json')
LOG_LEVEL = os.environ.get("LOG_LEVEL", logging.INFO)

import itertools
import numpy as np

def null_prep(vector):
    return vector

def exponential(vector):
    return [np.exp(v) for v in vector]

def step(vector):
    return [v if v > 0.5 else 0.0 for v in vector]

def quadratic(vector):
    return [v * v for v in vector]

def parent_children_funcs(parent, children, prep=null_prep):
    def custom_agg(vector):
        vector = prep(vector)
        if len(vector) > 1:
            return parent([vector[0], children(vector[1:])])
        else:
            return vector[0]
    return custom_agg

def build_combo_funcs(prep=null_prep):
    base_funcs = [np.mean, max]
    base_funcs.extend([parent_children_funcs(func_1, func_2, prep=prep) for (func_1, func_2) in itertools.permutations(base_funcs, 2)])
    return base_funcs

def combo_func_labels():
    base_funcs = ['np.mean', 'max']
    base_funcs.extend([func_1 + '+' + func_2 for (func_1, func_2) in itertools.permutations(base_funcs, 2)])
    return base_funcs
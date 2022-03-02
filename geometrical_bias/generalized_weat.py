"""
This script implements the Generalized Word Embedding Association Test (WEAT) from the paper
"What are the Biases in My Word Embedding?" by Swinger et. al (https://dl.acm.org/doi/pdf/10.1145/3306618.3314270).
"""

import numpy as np
from geometrical_bias import GeometricBias, EmbSetList, EmbSet, cossim

# TODO: adapt these funcions for multiclass WEAT
def s_wAB(w, A, B):
    return np.mean([cossim(w, a) for a in A]) - np.mean([cossim(w, b) for b in B])


def s_WAB(W, A, B):
    return np.sum([s_wAB(w, A, B) for w in W])


def s_XYAB(X, Y, A, B):
    return s_WAB(X, A, B) - s_WAB(Y, A, B)


def mean_s_wAB(W, A, B):
    return np.mean([s_wAB(w, A, B) for w in W])


def stdev_s_wAB(W, A, B):
    return np.std([s_wAB(w, A, B) for w in W], ddof=1)


# TODO multiclass WEAT effect size
def effect_size(target_groups, attribute_sets):
    print("not implemented yet")


# TODO
def permutation_test(target_groups, attribute_sets):
    print("not implemented yet")


class GeneralizedWEAT(GeometricBias):

    def __init__(self, *args, **kwargs):
        self.A = None

    def define_bias_space(self, attribute_sets: EmbSetList):
        assert len(attribute_sets) >= 2, "need at least two attribute groups to measure bias!"
        self.A = attribute_sets

    # TODO
    def individual_bias(self, target: np.ndarray):
        print("not implemented yet")

    def mean_individual_bias(self, targets: EmbSet):
        print("mean bias is not implement for generalized WEAT")
        pass

    def group_bias(self, target_groups: EmbSetList):
        assert len(target_groups) == len(self.A), "number of target groups must match the number of attribute sets!"
        return effect_size(target_groups, self.A)





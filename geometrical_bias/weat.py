"""
This script implements the Word Embedding Association Test (WEAT) from the paper
"Semantics derived automatically from language corpora contain human-like biases"
by Caliskan et. al (https://www.science.org/doi/full/10.1126/science.aal4230).
"""

import numpy as np
from geometrical_bias import GeometricBias, EmbSetList, EmbSet, cossim


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


def effect_size(X, Y, A, B):
    numerator = mean_s_wAB(X, A, B) - mean_s_wAB(Y, A, B)
    W = np.vstack([X,Y])
    denominator = stdev_s_wAB(W, A, B)
    return numerator / denominator


# TODO
def permutation_test(X, Y, A, B):
    print("not implemented yet")


class WEAT(GeometricBias):

    def __init__(self, *args, **kwargs):
        self.A = None

    def define_bias_space(self, attribute_sets: EmbSetList):
        assert len(attribute_sets) == 2, "WEAT needs exactly 2 attribute sets, use GeneralizedWEAT for more " \
                                         "than 2 protected groups"
        self.A = attribute_sets

    def individual_bias(self, target: np.ndarray):
        return s_wAB(target, self.A[0], self.A[1])

    def mean_individual_bias(self, targets: EmbSet):
        print("mean bias is not implement for WEAT")
        pass

    def group_bias(self, target_groups: EmbSetList):
        assert len(target_groups) == 2, "WEAT needs exactly 2 target groups, matching the number of attribute sets"
        return effect_size(target_groups[0], target_groups[1], self.A[0], self.A[1])





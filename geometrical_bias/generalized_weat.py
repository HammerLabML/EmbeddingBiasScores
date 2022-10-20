"""
This script implements the Generalized Word Embedding Association Test (WEAT) from the paper
"What are the Biases in My Word Embedding?" by Swinger et. al (https://dl.acm.org/doi/pdf/10.1145/3306618.3314270).
"""

import numpy as np
from geometrical_bias import GeometricBias, EmbSetList, EmbSet, cossim


class GeneralizedWEAT(GeometricBias):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_A = None
        self.mean_A_i = []

    def normalize_vectors(self, emb_sets: EmbSetList):
        normalized_sets = []
        for i in range(self.n):
            nvecs = []
            for emb in emb_sets[i]:
                nvecs.append(emb / np.linalg.norm(emb))
            normalized_sets.append(nvecs)
        return normalized_sets

    def define_bias_space(self, attribute_sets: EmbSetList):
        self.n = len(attribute_sets)
        assert self.n >= 2, "need at least two attribute groups to measure bias!"

        # normalize all attribute vectors
        self.A = self.normalize_vectors(attribute_sets)
        self.mean_A_i = [np.mean(self.A[i], axis=0) for i in range(self.n)]
        self.mean_A = np.mean(self.mean_A_i, axis=0)

    def individual_bias(self, target: np.ndarray):
        # The paper does not define the bias of a single word. Given the associated group of a single word,
        # one could (similarly to WEAT) compare the word vector with the "group vector" mean(A_i) - mean(A). This
        # does not exactly match the intuition behind the Generalized WEAT as introduced in the paper, because
        # we cannot contrast with words associated to the other groups, but is as close as we can get for one word.
        # If we don't know the group, one could compare the word to all groups and report the maximum bias:
        biases = [np.inner(target, self.mean_A_i[i] - self.mean_A) for i in range(self.n)]
        return np.max(biases)

    def mean_individual_bias(self, targets: EmbSet):
        print("mean bias is not implement for generalized WEAT")
        pass

    def group_bias(self, target_groups: EmbSetList):
        assert self.n >= 2, "need at least two attribute groups to measure bias!\ncall " \
                            "define_bias_space(attribute_sets) with a sufficient amount of attribute sets"
        assert len(target_groups) == self.n, "number of target groups must match the number of attribute sets!"

        # normalize all vectors
        X = self.normalize_vectors(target_groups)
        mean_X = np.mean([np.mean(Xi, axis=0) for Xi in X], axis=0)
        bias = [np.inner(np.mean(X[i], axis=0)-mean_X, self.mean_A_i[i]-self.mean_A) for i in range(self.n)]
        return np.sum(bias)





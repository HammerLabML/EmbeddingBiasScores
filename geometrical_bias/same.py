"""
This script implements the SAME score as described in the paper
"The SAME score: Improved cosine based bias score for word embeddings"
"""
import numpy as np
from geometrical_bias import GeometricBias, EmbSetList, EmbSet, normalize, cossim


class SAME(GeometricBias):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs = None

    def attr_mean(self, attribute_set):
        A_unit = normalize(attribute_set)
        center = np.mean(A_unit, axis=0)
        return center

    def attr_pair_diff(self, A, B):
        diff = self.attr_mean(A) - self.attr_mean(B)
        return diff

    def define_bias_space(self, attribute_sets: EmbSetList):
        self.n = len(attribute_sets)
        assert self.n >= 2, "need at least two attribute groups to measure bias!"
        self.A = attribute_sets

        # compute pairwise bias directions in advance
        self.pairs = [self.attr_pair_diff(attribute_sets[1], attribute_sets[0])]

        for i in range(2, self.n):
            # remove correlation with previous bias directions (otherwise we might observe the same bias twice)
            # we only need bias directions of pairs (attribute_sets[i], attribute_sets[0]), because others can be
            # written as a linear combination of these
            new_pair = self.attr_pair_diff(attribute_sets[i], attribute_sets[0])
            for pair in self.pairs:
                pair = pair / np.linalg.norm(pair)
                corr = np.dot(new_pair, pair)
                v_corr = corr * pair
                new_pair = new_pair - v_corr
            self.pairs.append(new_pair)

    def pair_bias(self, target, pair_diff):
        return cossim(target, pair_diff)

    def signed_individual_bias(self, target: np.ndarray):
        assert self.n == 2, "signed bias can only be obtained for exactly 2 groups"

        w = target / np.linalg.norm(target)  # need unit vectors!
        return self.pair_bias(w, self.pairs[0])

    def individual_bias(self, target: np.ndarray):
        assert self.n >= 2

        w = target / np.linalg.norm(target)  # need unit vectors!

        # initialize bias with bias to first pair
        bias = abs(self.pair_bias(w, self.pairs[0]))
        if len(self.pairs) == 1:
            return bias

        for i in range(1, len(self.pairs)):
            bias += abs(self.pair_bias(w, self.pairs[i])*np.linalg.norm(self.pairs[i]))

        return bias

    def mean_individual_bias(self, targets: EmbSet):
        return np.mean([self.individual_bias(target) for target in targets])
    
    def individual_bias_per_pair(self, target: np.ndarray):
        assert self.n >= 2

        w = target / np.linalg.norm(target)  # need unit vectors!
        
        biases = []
        biases.append(self.pair_bias(w, self.pairs[0]))
        
        for i in range(1, len(self.pairs)):
            biases.append(self.pair_bias(w, self.pairs[i]))
        return biases

    def individual_bias_pairwise(self, target: np.ndarray, group1: int, group2: int):
        assert (0 <= group1 < self.n and 0 <= group2 < self.n and not group1 == group2)
        pair_diff = self.attr_pair_diff(self.A[group1], self.A[group2])
        return self.pair_bias(target, pair_diff)

    def skew_pairwise(self, targets: EmbSet, group1: int, group2: int):
        assert (0 <= group1 < self.n and 0 <= group2 < self.n and not group1 == group2)
        pair_diff = self.attr_pair_diff(self.A[group1], self.A[group2])
        return np.mean([self.pair_bias(w, pair_diff) for w in targets])

    def stereotype_pairwise(self, targets: EmbSet, group1: int, group2: int):
        assert (0 <= group1 < self.n and 0 <= group2 < self.n and not group1 == group2)
        pair_diff = self.attr_pair_diff(self.A[group1], self.A[group2])
        return np.std([self.pair_bias(w, pair_diff) for w in targets])

    def group_bias(self, target_groups: EmbSetList):
        print("group bias is not implemented for SAME")
        pass

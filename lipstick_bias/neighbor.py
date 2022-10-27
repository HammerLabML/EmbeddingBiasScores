"""
This script implements the bias-by-neighbors test from "Lipstick on a Pig: Debiasing Methods Cover up Systematic
Gender Biases in Word Embeddings But do not Remove Them" by Gonen and Goldberg.
The bias_by_neighbor function implements the bias-by-neighbor as presented in the paper, other functions implement
an individual, mean and group bias similar to the bias-by-neighbor method but adapted to report biases independet of
stereotype assumptions.
"""
import numpy as np
from geometrical_bias import EmbSetList, EmbSet
from lipstick_bias import BiasGroupTest


class NeighborTest(BiasGroupTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # k neighbors are used to determine the bias of a word
        self.k = 100 if 'k' not in kwargs else kwargs.pop('k')

    # gives the groups of the k closest neighbors of target from the embeddings in X
    def closest_neighbor_groups(self, X: EmbSet, y: list, target: np.ndarray):
        distances = []
        for x in X:
            if (x == target).any():  # we do not want to compare x to itself if also in self.X
                distances.append(0)
            else:
                distances.append(np.linalg.norm(x-target))

        sort_idx = np.argsort(distances)
        top_k_groups = []
        k = min(self.k, len(sort_idx))
        for idx in sort_idx[:k]:
            top_k_groups.append(y[idx])
        return top_k_groups

    def define_bias_space(self, attribute_sets: EmbSetList):
        super().define_bias_space(attribute_sets)

    # calculate the percentage of neighbors per groups
    # the bias is the normalized difference of the highest group percentage and 1/n
    def individual_bias(self, target: np.ndarray):
        top_k_groups = self.closest_neighbor_groups(self.X, self.y, target)
        group_probs = []
        for i in range(self.n):
            group_probs.append(top_k_groups.count(i)/len(top_k_groups))
        print(group_probs)
        return (max(group_probs)-1.0/self.n)/(1-1.0/self.n)

    def mean_individual_bias(self, targets: EmbSet):
        return np.mean([self.individual_bias(target) for target in targets])

    def group_bias(self, target_groups: EmbSetList):
        print("group bias is not implemented for yet")
        pass

    # this implements the bias-by-neighbor as introduced in the paper
    def bias_by_neighbor(self, target_groups: EmbSetList):
        n = len(target_groups)
        assert n >= 2, "need at least two target groups to measure bias!"
        X = target_groups[0]
        y = [0] * len(target_groups[0])
        for i in range(n):
            X = np.vstack([X, target_groups[i]])
            y += [i] * len(target_groups[i])

        biases = []
        for i, group in enumerate(target_groups):
            for target in group:
                top_k_groups = self.closest_neighbor_groups(X, y, target)
                # percentage of words from the same group among the k nearest neighbors
                own_group_prob = top_k_groups.count(i) / len(top_k_groups)
                biases.append(own_group_prob)
        return biases

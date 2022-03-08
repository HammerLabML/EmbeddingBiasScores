"""
This script implements the Direct Bias from the paper "Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings" from Bolukbasi et al.
(https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf).
"""

import numpy as np
from sklearn.decomposition import PCA
from geometrical_bias import GeometricBias, EmbSetList, EmbSet, cossim


def sub_mean(pairs):
    means = np.mean(pairs, axis=0)
    for i, _ in enumerate(means):
        for j in range(len(pairs)):
            pairs[j][i, :] = pairs[j][i, :] - means[i]
    return pairs


def flatten(pairs):
    flattened = []
    for pair in pairs:
        for vec in pair:
            flattened.append(vec)
    return flattened


class DirectBias(GeometricBias):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias_space = None
        self.pca = None

        # as defined in the paper, c determines the strictness of bias measurement
        self.c = 1 if 'c' not in kwargs else kwargs.pop('c')

        # k is the number of dimensions for the bias subspace
        self.k = 1 if 'k' not in kwargs else kwargs.pop('k')

    def pca_bias_subspace(self):
        assert (self.n < self.A[0].shape[0]), "the number of samples should be larger " \
                                                   "than the number of bias attributes"
        encoded_pairs = sub_mean(np.copy(self.A))  # use a copy of the attributes!
        flattened = flatten(encoded_pairs)
        self.pca = PCA()
        self.pca.fit(flattened)
        return self.pca.components_[:self.k]

    def get_pca_explained_variance(self):
        return self.pca.explained_variance_[:self.k]

    def define_bias_space(self, attribute_sets: EmbSetList):
        self.n = len(attribute_sets)
        assert self.n >= 2, "need at least two attribute groups to measure bias!"
        self.A = attribute_sets
        self.bias_space = self.pca_bias_subspace()

    def individual_bias(self, target: np.ndarray):
        return abs(pow(np.sum([abs(cossim(target, bias_dir)) for bias_dir in self.bias_space]), self.c))

    def mean_individual_bias(self, targets: EmbSet):
        return np.mean([self.individual_bias(target) for target in targets])

    def group_bias(self, target_groups: EmbSetList):
        print("group bias is not implemented for DirectBias")
        pass

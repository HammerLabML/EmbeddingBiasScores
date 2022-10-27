"""
This script implements the bias-by-neighbors test from "Lipstick on a Pig: Debiasing Methods Cover up Systematic
Gender Biases in Word Embeddings But do not Remove Them" by Gonen and Goldberg.
"""
import numpy as np
from scipy import spatial
from geometrical_bias import GeometricBias, EmbSetList, EmbSet


class BiasGroupTest(GeometricBias):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = None
        self.y = None

    # attributes can be any terms sorted into groups of stereotypes
    # they can be used as reference for new words, where the stereotype is not known
    def define_bias_space(self, attribute_sets: EmbSetList):
        self.n = len(attribute_sets)
        assert self.n >= 2, "need at least two attribute groups to measure bias!"
        self.A = attribute_sets

        self.X = self.A[0]
        self.y = [0] * len(self.A[0])
        for i in range(1, self.n):
            self.X = np.vstack([self.X, self.A[i]])
            self.y += [i] * len(self.A[i])

    def individual_bias(self, target: np.ndarray):
        pass

    def mean_individual_bias(self, targets: EmbSet):
        pass

    def group_bias(self, target_groups: EmbSetList):
        pass

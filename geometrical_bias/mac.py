"""
This script implements the MAC score from the paper "Black is to Criminal as Caucasian is to Police:
Detecting and Removing Multiclass Bias in Word Embeddings" by Manzini et. al (https://aclanthology.org/N19-1062.pdf).
"""
import numpy as np
from scipy import spatial
from geometrical_bias import GeometricBias, EmbSetList, EmbSet


class MAC(GeometricBias):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def define_bias_space(self, attribute_sets: EmbSetList):
        self.n = len(attribute_sets)
        assert self.n >= 2, "need at least two attribute groups to measure bias!"
        self.A = attribute_sets

    def individual_bias(self, target: np.ndarray):
        return np.mean([spatial.distance.cosine(target, a) for A_j in self.A for a in A_j])

    def mean_individual_bias(self, targets: EmbSet):
        return np.mean([self.individual_bias(target) for target in targets])

    def group_bias(self, target_groups: EmbSetList):
        print("group bias is not implemented for MAC")
        pass

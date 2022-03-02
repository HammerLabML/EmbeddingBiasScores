import numpy as np
from typing import List
import math

EmbSetList = List[List[np.ndarray]]
EmbSet = List[np.ndarray]

def normalize(vectors: np.ndarray):
    norms = np.apply_along_axis(np.linalg.norm, 1, vectors)
    vectors = vectors / norms[:, np.newaxis]
    return np.asarray(vectors)


def cossim(x: np.ndarray, y: np.ndarray):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


class GeometricBias:

    def __init__(self, *args, **kwargs):
        self.verbose = False if 'verbose' not in kwargs else kwargs.pop('verbose')

    def define_bias_space(self, attribute_sets: EmbSetList):
        pass

    def individual_bias(self, target: np.ndarray) -> float:
        pass

    def mean_individual_bias(self, targets: EmbSet) -> float:
        pass

    def group_bias(self, target_groups: EmbSetList) -> float:
        pass

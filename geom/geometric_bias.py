import numpy as np

EmbSetList = list[list[np.ndarray]]
EmbSet = list[np.ndarray]


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

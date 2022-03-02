"""
This script implements the relational inner product association (RIPA) as introduced by Ethayarajh et al.
in their paper "Understanding Undesirable Word Embedding Associations" (https://aclanthology.org/P19-1166.pdf)
"""
import numpy as np
from geometrical_bias import EmbSetList, DirectBias


class RIPA(DirectBias):

    def individual_bias(self, target: np.ndarray):
        return abs(pow(np.sum([abs(np.inner(target, bias_dir)) for bias_dir in self.bias_space]), self.c))

    def group_bias(self, target_groups: EmbSetList):
        print("group bias is not implemented for RIPA")
        pass

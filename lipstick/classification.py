"""
This script implements the classification test from "Lipstick on a Pig: Debiasing Methods Cover up Systematic
Gender Biases in Word Embeddings But do not Remove Them" by Gonen and Goldberg.

"""
import numpy as np
from geometrical_bias import EmbSetList, EmbSet
from lipstick import BiasGroupTest
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


class ClassificationTest(BiasGroupTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.svc = None

    def define_bias_space(self, attribute_sets: EmbSetList):
        super.define_bias_space(attribute_sets)

        # TODO: train

    def individual_bias(self, target: np.ndarray):
        if self.svc is None:
            print("need to define the bias space first!")
            return

        print("individual bias is not implemented for yet")
        pass

    def mean_individual_bias(self, targets: EmbSet):
        return np.mean([self.individual_bias(target) for target in targets])

    def group_bias(self, target_groups: EmbSetList):
        print("group bias is not implemented for yet")
        pass

    # this implements the cluster test as introduced in the paper
    def classification_test(self, target_groups: EmbSetList, cv_folds=5):
        n = len(target_groups)
        X = target_groups[0]
        y = [0] * len(target_groups[0])
        for i in range(n):
            X = np.vstack([X, target_groups[i]])
            y += [i] * len(target_groups[i])

        self.svc = SVC(kernel='rbf')
        scores = cross_val_score(self.svc, X, y,  cv=cv_folds)
        return scores

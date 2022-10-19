"""
This script implements the cluster test from "Lipstick on a Pig: Debiasing Methods Cover up Systematic
Gender Biases in Word Embeddings But do not Remove Them" by Gonen and Goldberg.
The cluster_test function implements the test as presented in the paper, other functions implement
an individual, mean and group bias similar to the cluster test method but adapted to report biases independet of
stereotype assumptions and for multiclass settings.
"""
import numpy as np
from geometrical_bias import EmbSetList, EmbSet
from lipstick import BiasGroupTest
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# TODO proper clustering metric for multiclass bias

class ClusterTest(BiasGroupTest):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kmeans = None
        self.pgroup_per_cluster = {}
        self.cluster_label = []

    def define_bias_space(self, attribute_sets: EmbSetList):
        super().define_bias_space(attribute_sets)

        self.kmeans = KMeans(n_clusters=self.n, random_state=0).fit(self.X)
        y_cluster = self.kmeans.predict(self.X)
        print("predicted cluster on defining data:")
        print(y_cluster)

        self.cluster_label = []
        for c in range(self.n):
            c_label = []
            for i in range(len(y_cluster)):
                if y_cluster[i] == c:
                    c_label.append(self.y[i])
            majority_class = 0
            count = 0
            for c2 in range(self.n):
                if c_label.count(c2) > count:
                    majority_class = c2
                    count = c_label.count(c2)
            self.cluster_label.append(majority_class)
        print("cluster label: ")
        print(self.cluster_label)

        print("accuracy on defining data:")
        acc = accuracy_score(self.y, y_cluster)
        # since cluster ids are ambiguous, we might need to take 1-accuracy
        if acc < 0.5:
            acc = 1 - acc
        print(acc)

    def individual_bias(self, target: np.ndarray):
        print("individual bias is not implemented")
        pass

    def mean_individual_bias(self, targets: EmbSet):
        print("mean individual bias is not implemented")
        pass

    def group_bias(self, target_groups: EmbSetList):
        print("group bias is not implemented for yet")
        pass

    # this implements the cluster test as introduced in the paper
    def cluster_test(self, target_groups: EmbSetList, cv_folds=5):
        n = len(target_groups)
        assert n == 2, "need exactly two target groups!"
        X = target_groups[0]
        y = [0] * len(target_groups[0])
        for i in range(n):
            X = np.vstack([X, target_groups[i]])
            y += [i] * len(target_groups[i])

        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        y_pred = kmeans.predict(X)
        acc = accuracy_score(y, y_pred)

        if acc < 0.5:
            acc = 1 - acc

        return acc



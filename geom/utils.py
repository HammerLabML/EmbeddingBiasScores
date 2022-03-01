import numpy as np
import math


def normalize(vectors: np.ndarray):
    norms = np.apply_along_axis(np.linalg.norm, 1, vectors)
    vectors = vectors / norms[:, np.newaxis]
    return np.asarray(vectors)


def cossim(x: np.ndarray, y: np.ndarray):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))


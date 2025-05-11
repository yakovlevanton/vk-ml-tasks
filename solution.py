import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    diag = matrix.diagonal()
    nonzero_elems = diag[diag != 0]
    return np.prod(nonzero_elems)


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    return np.array_equal(np.unique(x), np.unique(y))


def max_before_zero_vectorized(x: np.array):
    true_indices = np.where(x[:-1] == 0)[0] + 1
    return np.max(x[true_indices])


def add_weighted_channels_vectorized(image: np.array):
    weights = np.array([0.299, 0.587, 0.114])
    image = np.dot(image, weights)
    return image


def run_length_encoding_vectorized(x: np.array):
    borders = np.concatenate(([0], np.where(x[:-1] != x[1:])[0] + 1, [x.size]))
    uniqs = x[borders[:-1]]
    lengths = np.diff(borders)
    return uniqs, lengths

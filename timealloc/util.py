import numpy as np


def fill_from_2d_array(array):
    """
    Pyomo is 1-index and does not natively support multi-dimensional numpy
    arrays, so this function can be used to initialize parameters/sets from a
    zero-indexed 2D numpy array

    Usage: self.model.array = Param(size, initialize=fill_from_array(array))


    Here's another way to do it:
     m.D = Param(m.N, m.N, initialize=dict(((i,j),D[i,j]) for i in m.N for j
     in m.N))
     Source: https://stackoverflow.com/a/41834158

    :param array: 2D numpy array
    :return: initialization function
    """
    def fn(model, i, j):
        return array[i-1, j-1]
    return fn


def fill_from_array(array):
    def fn(model, i):
        return array[i-1]
    return fn


def linop_from_1d_filter(filter, n, offset=0):
    L = np.zeros((n - filter.size + 1 + offset * 2, n))
    i, j = np.indices(L.shape)
    for k in range(filter.size):
        L[i == j - k + offset] = filter[k]
    bias = np.zeros(L.shape[0])
    for k in range(offset):
        bias[k] = offset - k
        bias[-k - 1] = offset - k
    return L, bias

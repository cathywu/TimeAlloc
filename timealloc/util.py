import numpy as np


def fill_from_2d_array(array):
    """
    Pyomo does not natively support multi-dimensional numpy arrays, so this
    function can be used to initialize parameters/sets from a zero-indexed 2D
    numpy array.

    Usage: model.array = Param(IndexSet, initialize=fill_from_array(array))

    Here's another way to do it:
     m.D = Param(m.N, m.N, initialize=dict(((i,j),D[i,j]) for i in m.N for j
     in m.N))
     Source: https://stackoverflow.com/a/41834158

    :param array: 2D numpy array
    :return: initialization function
    """

    def fn(model, i, j):
        return array[i, j]

    return fn


def fill_from_array(array):
    """
    Returns a function for initializing a Pyomo object with a numpy array.

    Usage: model.array = Param(IndexSet, initialize=fill_from_array(array))

    :param array: 1D numpy array
    :return: initialization function
    """

    def fn(model, i):
        return array[i]

    return fn


def linop_from_1d_filter(filter, n, offset=0, offset_end=None):
    """
    Converts a 1D filter (numpy array) to a 1-shifted linear operator,
    which can be applied to a vector (or matrix).

    Usage:
     filter = np.array([-1, 1, -1])
     L, b = linop_from_1d_filter(filter, size, offset=1)

    :param filter: 1d numpy array
    :param n: dimensionality of input space
    :param offset: determines start offset
    :param offset_end: determines end offset
    :return: m-by-n linear operator, where m is calculated via n, the filter
             length and offsets
    """
    if offset_end is not None:
        L = np.zeros((n - filter.size + 1 + offset + offset_end, n))
    else:
        L = np.zeros((n - filter.size + 1 + offset * 2, n))
    i, j = np.indices(L.shape)
    for k in range(filter.size):
        L[i == j - k + offset] = filter[k]
    bias = np.zeros(L.shape[0])
    for k in range(offset):
        bias[k] = offset - k
        bias[-k - 1] = offset - k
    return L, bias


def triu(m, incr=1):
    """
    Returns upper triangular matrix with stride length incr

    Example output with m=6, incr=2
    0 0 1 1 1 1
    0 0 0 0 1 1

    :param m: range space dimension
    :param incr: increment/offset for starting the next row
    :return: m/incr-by-m matrix
    """
    n = int(m / incr)-1
    L = np.zeros((n, m))
    for i in range(n):
        L[i, (i+1) * incr:] = 1
    return L


def tril(m, incr=1):
    """
    Returns lower triangular matrix with stride length incr

    Example output with m=6, incr=2
    1 1 0 0 0 0
    1 1 1 1 0 0

    :param m: range space dimension
    :param incr: increment/offset for ending the next row
    :return: m/incr-by-m matrix
    """
    n = int(m / incr)-1
    L = np.zeros((n, m))
    for i in range(n):
        L[i, :(i + 1) * incr] = 1
    return L


def html_from_md(fname):
    """
    Reads in a file as markdown, converts it to html, and then returns the
    html soup
    :param fname: Markdown filename
    :return:
    """
    from bs4 import BeautifulSoup
    import mistune
    inp = open(fname).read()
    html = mistune.markdown(inp)
    soup = BeautifulSoup(html, features="html.parser")
    return soup

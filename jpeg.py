from scipy import fftpack
import numpy as np
from ml_util import ml


dct = lambda x, axis=1: fftpack.dct(x, type=2, axis=axis)
idct = lambda x, axis=1: fftpack.idct(x, type=2, axis=axis)

def jpeg_compress(X, Q):
    assert(X.shape == Q.shape)

    one_d = np.reshape(X - 127, -1).astype(float)
    dcted = np.reshape(dct(one_d), X.shape)
    quantized = np.round(dcted / Q)

    return quantized

def truncate(x, d):
    assert(ml.isVector(x))
    D = x.shape[0]

    if d >= D:
    	return x

    ret = np.compress([True for _ in xrange(d)], x, axis=0)
    assert(ret.shape == (d,))
    return ret

def truncate_batch(X, d):
    assert(ml.dim(X) == 2)
    N, D = X.shape

    if d >= D:
        d = D

    ret = np.compress([True for _ in xrange(d)], X, axis=1)
    assert(ret.shape == (N, d))


# Transforms a bit array with values between 0-1 to
# one with values between 0-k
def normal_to_bitmap(X, k=256): return np.rint(X * k)

def bitmap_to_normal(X, k=256): return X / float(k)

# Runs DCT on a two-dimensional image
def dct_truncate(X, n, height, width):
    two_d = np.reshape(X, -1).astype(float).reshape((height, width)) - 127
    dcted = dct(dct(two_d, axis=0), axis=1)
    return truncate(np.reshape(dcted, X.shape), n)


def jpeg_decompress(Y, Q):
    assert(Y.shape == Q.shape)

    dcted = Y * Q
    X = np.reshape(idct(np.reshape(dcted, -1)), Y.shape)
    return np.round(X + 127)


# X = np.array([[1, 2, 3, 0], [4, 5, 6, 7], [8 ,9, 10, 11]])
# X1 = np.array([[0, 0, 255, 0], [0, 256, 0, 0], [0 ,250, 0, 0]])

# Q = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2]])

# Xp = np.reshape(X, -1)

# print X
# print jpeg_compress(X, Q)

# print X1
# print jpeg_compress(X1, Q)

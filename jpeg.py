from scipy import fftpack
import numpy as np


dct = lambda x: fftpack.dct(x, norm='ortho')
idct = lambda x: fftpack.idct(x, norm='ortho')

def jpeg_compress(X, Q):
    assert(X.shape == Q.shape)

    one_d = np.reshape(X - 127, -1).astype(float)
    dcted = np.reshape(dct(one_d), X.shape)
    quantized = np.round(dcted / Q)

    return quantized

def truncate(x, n):
    if n >= x.shape[0]:
    	n = x.shape[0]

    r = []
    for i in xrange(n):
    	r.append(x[i])
    return np.array(r)

def dct_truncate(X, n):
    one_d = np.reshape(X - 127, -1).astype(float)
    dcted = np.reshape(dct(one_d), X.shape)
    return truncate(dcted, n)


def jpeg_decompress(Y, Q):
    assert(Y.shape == Q.shape)

    dcted = Y * Q
    X = np.reshape(idct(np.reshape(dcted, -1)), Y.shape)
    return np.round(X + 127)

X = np.array([[1, 2, 3, 0], [4, 5, 6, 7], [8 ,9, 10, 11]])

Q = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2]])

Xp = np.reshape(X, -1)

print dct_truncate(Xp, 100)
print dct_truncate(Xp, 5)

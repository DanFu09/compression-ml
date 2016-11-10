from scipy import fftpack
import numpy as np


dct = lambda x: fftpack.dct(x, norm='ortho')
idct = lambda x: fftpack.idct(x, norm='ortho')


X = np.array([[1, 2, 3, 0], [4, 5, 6, 7], [8 ,9, 10, 11]])

Q = np.array([[2,2,2,2],[2,2,2,2],[2,2,2,2]])

def jpeg_compress(X, Q):
    assert(X.shape == Q.shape)

    one_d = np.reshape(X - 127, -1).astype(float)
    dcted = np.reshape(dct(one_d), X.shape)
    quantized = np.round(dcted / Q)

    return quantized


def jpeg_decompress(Y, Q):
    assert(Y.shape == Q.shape)

    dcted = Y * Q
    X = np.reshape(idct(np.reshape(dcted, -1)), Y.shape)
    return np.round(X + 127)


print X
print jpeg_decompress(jpeg_compress(X, Q), Q)


real_X = np.array([[124, 125, 122, 120, 122, 119, 117, 118],
    [120, 120, 120, 119, 119, 120, 120, 120],
    [125, 124, 123, 122, 121, 120, 119, 118],
    [125, 124, 123, 122, 121, 120, 119, 118],
    [130, 131, 132, 133, 134, 130, 126, 122],
    [140, 137, 137, 133, 133, 137, 135, 130],
    [150, 147, 150, 150, 150, 150, 150, 150],
    [160, 160, 162, 164, 168, 170, 172, 175]])

real_Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
    [12 ,12 , 14 , 19 , 26 , 58 , 60 , 55],
    [14 ,13 , 16 , 24 , 40 , 57 , 69 , 56],
    [14 ,17 , 22 , 29 , 51 , 87 , 80 , 62],
    [18 ,22 , 37 , 56 , 68 , 109 , 103 , 77],
    [24 ,35 , 55 , 64 , 81 , 104 , 113 , 92],
    [49 ,64 , 78 , 87 , 103 , 121 , 120 , 101],
    [72 ,92 , 95 , 98 , 112 , 100 , 103 , 99]])

print real_X
compressed = jpeg_compress(real_X, real_Q)
print compressed
print jpeg_decompress(compressed, real_Q)

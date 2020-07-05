import numpy as np
from scipy import ndimage, misc


def B_k(b, k, Ih, u, v):
    if (2 * k * np.pi) / b <= Ih[u, v] < (2 * (k+1) * np.pi) / b:
        bk = 1
    else:
        bk = 0
    return bk


def sum_arg(b, k, Ih, r, N):
    w = Ih.shape[1]
    h = Ih.shape[0]
    Bk = np.zeros((h,w))
    for u in range(h):
        for v in range(w):
            Bk[u, v] = B_k(b, k, Ih, u, v)
    comp_1 = ndimage.gaussian_filter(Bk, sigma=r/3)
    comp_2 = np.log2(comp_1 + 1e-16)
    nominator = np.multiply(comp_1, comp_2)
    denominator = np.log2(N)
    return - nominator/denominator


def extract_feat2(Is, Ih, r, N=3):
    beta = 0.25
    b = 10
    k = 0
    comp_1 = np.zeros(sum_arg(b, k, Ih, r, N).shape)
    for k in range(b+1):
        comp_1 += sum_arg(b, k, Ih, r, N)
    comp_2 = np.power(ndimage.gaussian_filter(Is, sigma=r/3), beta)
    feat = np.multiply(comp_1, comp_2)

    return feat




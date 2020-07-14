import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
from skimage import data,io

from skimage.color import rgb2hsv
from scipy import ndimage, misc
from skimage.feature import hog
from skimage import data, exposure
from sklearn.cluster import KMeans

def extract_feat3(Iv,r):
    #we change the pixel per cell from 1*1 to 16*16 based on Dalal and triggs
    fd, hog_image = hog(Iv, orientations=8, pixels_per_cell=(16,16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    w = Iv.shape[1]
    h = Iv.shape[0]
    res = ndimage.gaussian_filter(hog_image,sigma=r/3)
    return res

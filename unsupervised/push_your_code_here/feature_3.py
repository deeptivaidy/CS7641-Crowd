import numpy as np
from scipy import ndimage, misc

from skimage.feature import hog
from skimage import data, exposure

def extract_feat3(Iv,r):
    #we change the pixel per cell from 1*1 to 16*16 based on Dalal and triggs
    fd, hog_image = hog(Iv, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)
    w = Iv.shape[1]
    h = Iv.shape[0]
    res = np.zeros((h,w))
    for i in range(0,h,16):
        for j in range(0,w,16):
            feat = ndimage.gaussian_filter(fd[i:i+8], sigma=r/3)
            feat = np.sqrt(np.sum(feat**2))
            if(i + 16 <h and j + 16<w):
                   temp = np.zeros([16,16])
                   temp.fill(feat)
                   res[i:i+16,j:j+16] = temp
            elif(i + 16 >= h and j + 16 < w):
                   temp = np.zeros([h-i,16])
                   temp.fill(feat)
                   res[i:h,j:j+16] = temp
            elif(i + 16 < h and j + 16 >= w):
                   temp = np.zeros([16,w-j])
                   temp.fill(feat)
                   res[i:i+16,j:w] = temp
            elif(i + 16 >= h and j + 16 >= w):
                   temp = np.zeros([h-i,w-j])
                   temp.fill(feat)
                   res[i:h,j:w] = temp
    return res
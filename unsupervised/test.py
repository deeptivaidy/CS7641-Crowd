import matplotlib.pyplot as plt

import numpy as np
from skimage import data, io

from skimage.color import rgb2hsv
from scipy import ndimage, misc
from Util.extract_feat1 import extract_feat1
from Util.extract_feat2 import extract_feat2
from Util.extract_feat3 import extract_feat3


rgb_img = io.imread('test_img/test1.jpg')
hsv_img = rgb2hsv(rgb_img)
hue_img = hsv_img[:, :, 0]
sat_img = hsv_img[:, :, 1]
value_img = hsv_img[:, :, 2]

'''
fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(16, 4))

ax0.imshow(rgb_img)
ax0.set_title("RGB image")
ax0.axis('off')
ax1.imshow(hue_img, cmap='hsv')
ax1.set_title("Hue channel")
ax1.axis('off')
ax2.imshow(sat_img, cmap='hsv')
ax2.set_title("Saturation channel")
ax2.axis('off')
ax3.imshow(value_img)
ax3.set_title("Value channel")
ax3.axis('off')

fig.tight_layout()
plt.show()
'''





feat2 = extract_feat2(sat_img, hue_img, 2)
f, axarr = plt.subplots(1,2, figsize=(8, 3))
fig1 = axarr[0].imshow(rgb_img)
axarr[0].set_title("Input Image")
fig2 = axarr[1].imshow(feat2)
fig2.set_cmap('Greys')
axarr[1].set_title('Entropy')
plt.show()

feat3 = extract_feat3(value_img, 2)
f, axarr = plt.subplots(1,2, figsize=(8, 3))
fig1 = axarr[0].imshow(rgb_img)
axarr[0].set_title("Input Image")
fig2 = axarr[1].imshow(feat3)
fig1.set_cmap('Greys')
axarr[1].set_title('Histogram')
plt.show()

feat1 = extract_feat1(hue_img, sat_img, 2)
r = np.max(feat1) - np.min(feat1)
scale = 255/r
feat1_scaled = (feat1 - np.min(feat1)) * scale
f, axarr = plt.subplots(1,2, figsize=(8,3))
fig1 = axarr[0].imshow(rgb_img)
axarr[0].set_title("Input Image")
fig2 = axarr[1].imshow(feat1<0)
fig2.set_cmap("Greys")
axarr[1].set_title('LoG')
plt.show()






















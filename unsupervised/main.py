## -------import library-------
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage import data,io
from sklearn.cluster import KMeans
from utils import feature_extractor
import os
import time

# saving directory
save_dir = "results/"
data_dir = "test_img/selected_test_img/"

# image number
img_no = 52


## -------Load Image-------
# img: h x w x 3
# skimage.io.imread
img_dir = os.path.join(data_dir, "IMG_{}.jpg".format(img_no))
rgb_img = io.imread(img_dir)
plt.imshow(rgb_img)

## -------Extract Features-------
# feature_mat: h x w x n
# n: number of features per pixel (depends on number of window size r)
# e.g. r = range(1,50)
r = np.array([1])
# r = np.arange(10, 60, 10)   ##### Window size
feature_mat = feature_extractor(rgb_img, r)

## -------CLustering-------
# cluster_map: h x w (only contains 0 or 1; 0 for background; 1 for crowd)
# cluster_map = kmeans(feature_mat, k=2)
a, b, c = feature_mat.shape
input_data = np.reshape(feature_mat, (a*b, c))
t0 = time.time()
kmeans = KMeans(n_clusters=2, random_state=0).fit(input_data)
labels = kmeans.labels_
cluster_map = np.reshape(labels, (a,b))
t1 = time.time()
print("K-Means finished, used {} sec.".format(round(t1-t0, 2)))


## -------Plotting resulting image-------
pkl_dir = os.path.join(save_dir, "IMG_{}_kmeans.pkl".format(img_no))
f = open(pkl_dir, "wb")
pickle.dump([kmeans, cluster_map], f)
f.close()

plot_dir = os.path.join(save_dir, "IMG_{}_kmeans.jpg".format(img_no))
plt.imshow(cluster_map)
plt.savefig(plot_dir)
plt.show()

## -------Compare with ground truth-------

## -------import library-------
import matplotlib.pyplot as plt
import numpy as np
from skimage import data,io
from sklearn.cluster import KMeans
from utils import feature_extractor


## -------Load Image-------
# img: h x w x 3
# skimage.io.imread
rgb_img = io.imread('test_img/IMG_52.jpg')
plt.imshow(rgb_img)

## -------Extract Features-------
# feature_mat: h x w x n
# n: number of features per pixel (depends on number of window size r)
# e.g. r = range(1,50)
r = np.array([1,5]) #np.arange(10,60,10)
feature_mat = feature_extractor(rgb_img, r)

## -------CLustering-------
# cluster_map: h x w (only contains 0 or 1; 0 for background; 1 for crowd)
# cluster_map = kmeans(feature_mat, k=2)
a,b,c = feature_mat.shape
input_data = np.reshape(feature_mat, (a*b, c))
kmeans = KMeans(n_clusters=2, random_state=0).fit(input_data)
labels = kmeans.labels_
cluster_map = np.reshape(labels,(a,b))

## -------Plotting resulting image-------
plt.imshow(cluster_map)
plt.show()
plt.savefig('Results_52.jpg')

## -------Compare with ground truth-------

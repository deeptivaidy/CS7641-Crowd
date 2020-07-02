## -------import library-------


## -------Load Image-------
# img: h x w x 3
# skimage.io.imread

## -------Extract Features-------
# feature_mat: h x w x n
# n: number of features per pixel (depends on number of window size r)
# e.g. r = range(1,50)

# feature_mat = feature_extractor(img, r)

## -------CLustering-------
# cluster_map: h x w (only contains 0 or 1; 0 for background; 1 for crowd)

# cluster_map = kmeans(feature_mat, k=2)

## -------Plotting resulting image-------


## -------Compare with ground truth-------
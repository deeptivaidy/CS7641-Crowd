## import libs


## -------Extract feature 1 (LoG)-------
# feature1: h x w 


## -------Extract feature 2 (Entropy)-------
# feature2: h x w


## -------Extract feature 3 (HOG)-------
# feature3: h x w


## -------Feature Extractor-------
def feature_extractor(img, r):
    """ img: h x w x 3
       r: m dimensional vector
       feature_mat: h x w x 3m
    """
    
    ## Convert to HSV 
    # img_HSV: h x w x 3
    
    for i in r:
        ## feature1[:,:,i] = extract_feat1()

        ## feature2[:,:,i] = extract_feat2()

        ## feature3[:,:,i] = extract_feat3()
    
    ## concatenate feaeture1,2,3
    # feature_mat = 
    
    return feature_mat
    
    
    
## -------Clustering-------    
# cluster_map: h x w (only contains 0 or 1; 0 for background; 1 for crowd)

# cluster_map = kmeans(feature_mat, k=2)   
    
    
    
    
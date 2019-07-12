import numpy as np
import cv2
import os
from scipy import ndimage
from scipy.spatial import distance
from sklearn.cluster import KMeans

# takes two arrays as parameters and find the l1 distance
def L1_dist(vec1, vec2):
    return np.linalg.norm(np.subtract(np.array(vec1), np.array(vec2)))    
#This makes concatanetion of the images and wirte them
def images_print(images, feature_vectors, test_vectors, test):
    loop_num = 0
    samples_name = ["iris", "teapot", "elk"]
    for i in samples_name:
        cv2.imwrite(str(i)+".png", test[i][4])
        closest_images = closests(feature_vectors, test_vectors[i][4])
        x = []
        for ind in range(len(closest_images)):
            x.append(cv2.resize(images[closest_images[ind][0]][closest_images[ind][1]],(250,250))) 
        img_concatanete = np.concatenate((x[0],x[1],x[2],x[3],x[4]),axis=1)
        cv2.imwrite('the_closest_images_to_'+ str(i)+".png",img_concatanete)

# Returns the most 5 similar categories. 
# Takes 2 parameters image dictionary and test data
def closests(images, test):
    img = [["", 0], ["", 0], ["",0], ["",0], ["",0]]
    dist = [np.inf, np.inf, np.inf, np.inf, np.inf]
    
    for key, value in images.items():
        for ind in range(len(value)):
            dist_val = distance.euclidean(test, value[ind])
            #dist_val = L1_dist(test, value[ind])
            for i in range(len(dist)):
                if(dist_val < dist[i]):
                    dist[i] = dist_val
                    img[i][0] = key
                    img[i][1] = ind
                    break
    return img

# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each tiles of the images that are separated class by class. 
def image_class_tiling(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            tiles_hist = []
            for val in img:
                histogram = np.zeros(len(centers))
                for each_feature in val:
                    ind = find_index(each_feature, centers)
                    histogram[ind] += 1
               
                
                tiles_hist.extend(histogram)
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

# Creates descriptors using sift library for each tile
# Takes one parameter that is images dictionary that holds the tiles not the pictures itselves
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def sift_features_tiling(images):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        features = []
        for img in value:
            tiling = []
            for tile in img:
            
                kp, des = sift.detectAndCompute(tile,None)
                if(len(kp)>=1):

                    descriptor_list.extend(des)

                    tiling.append(des)
            features.append(tiling)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

# Helps the tiling function which finds one of the multipliers of the k value.
# Takes the k values as a parameter
# Returns the one of the multipliers of the k value. 
def find_multiplier(num):
    multiplier = 0
    if(num > 50):
        for i in range(10,50):
            if(num % i == 0):
                multiplier = i
                return multiplier
    else:
        for i in range(1,20):
            if(num % i == 0):
                multiplier = i
                return multiplier
    return multiplier

# split the image k pieces.
# Takes images dictionary and number of pieces
# Return a dictionary that holds the tiles of the images which are seperated class by class
def tiling(images, k):
    images_tiling = {}
    for key,value in images.items():
        image_cat = []
        for img in value:
            image = []
            
            width = img.shape[1]
            height = img.shape[0]
            
            multiplier_width = find_multiplier(k)
            if(multiplier_width != 0):
                multiplier_height = int(k / multiplier_width)
                width_step = int(np.floor(width / multiplier_width))
                height_step = int(np.floor(height / multiplier_height))
                start_width = 0
                end_width = width_step
                start_height = 0
                end_height = height_step
                for step_width in range(multiplier_width):
                    for step_height in range(multiplier_height):
                        tile = img[start_height:end_height,start_width:end_width]
                        image.append(tile)
                        start_height = end_height
                        end_height = start_height + height_step
                    start_width = end_width
                    end_width = start_width + width_step
                    start_height = 0
                    end_height = height_step


            else:
                resized = cv2.resize(img, (k, height), interpolation = cv2.INTER_AREA)
                width_step = 1
                start = 0
                end = width_step
                for step in range(k):
                    tile = resized[0:height,start:end]
                    start = end
                    end = start + width_step
                    image.append(tile)
            image_cat.append(image)
        images_tiling[key] = image_cat
    return images_tiling



# Find the index of the closest central point to the each sift descriptor. 
# Takes 2 parameters the first one is a sift descriptor and the second one is the array of central points in k means
# Returns the index of the closest central point.  
def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
           count = distance.euclidean(image, center[i]) 
           #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i]) 
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind

# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class. 
def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

# A k-means clustering algorithm who takes 2 parameter which is number of cluster(k) and the other is descriptors list(unordered 1d array)
# Returns an array that holds central points.
def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words

# Creates descriptors using sift library
# Takes one parameter that is images dictionary
# Return an array whose first index holds the decriptor_list without an order
# And the second index holds the sift_vectors dictionary which holds the descriptors but this is seperated class by class
def sift_features(images):
    sift_vectors = {}
    descriptor_list = []
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img,None)
           
            
            descriptor_list.extend(des)
            features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

# Create the sift feature vectors(1X128) for each image.
# Takes images as a parameter. It is the dictionary of images (class by class) whose features should be extracted
#  Return a dictianory that holds the features class by class
def sift_filters(images):
    sift_vectors = {}
    sift = cv2.xfeatures2d.SIFT_create()
    for key,value in images.items():
        features = []
        for img in value:
            kp, des = sift.detectAndCompute(img,None)
            features.append(des.mean(0)) # take the average and 1x128 matrix we get
        sift_vectors[key] = features
    return sift_vectors

# Calculates the average accuracy and class based accuracies.  
def accuracy(results):
    avg_accuracy = (results[1] / results[0]) * 100
    print("Average accuracy: %" + str(avg_accuracy))
    print("\nClass based accuracies: \n")
    for key,value in results[2].items():
        acc = (value[0] / value[1]) * 100
        print(key + " : %" + str(acc))

# 1-NN algorithm. We use this for predict the class of test images.
# Takes 2 parameters. images is the feature vectors of train images and tests is the feature vectors of test images
# Returns an array that holds number of test images, number of correctly predicted images and records of class based images respectively
def knn(images, tests):
    num_test = 0
    correct_predict = 0
    class_based = {}
    
    for test_key, test_val in tests.items():
        class_based[test_key] = [0, 0] # [correct, all]
        for tst in test_val:
            predict_start = 0
            #print(test_key)
            minimum = 0
            key = "a" #predicted
            for train_key, train_val in images.items():
                for train in train_val:
                    if(predict_start == 0):
                        minimum = distance.euclidean(tst, train)
                        #minimum = L1_dist(tst,train)
                        key = train_key
                        predict_start += 1
                    else:
                        dist = distance.euclidean(tst, train)
                        #dist = L1_dist(tst,train)
                        if(dist < minimum):
                            minimum = dist
                            key = train_key
            
            if(test_key == key):
                correct_predict += 1
                class_based[test_key][0] += 1
            num_test += 1
            class_based[test_key][1] += 1
            #print(minimum)
    return [num_test, correct_predict, class_based]           

# Creates the gabor feature vectors.
# Takes images and filters as parameters. images holds the train images and filters holds the all filters
# Returns the feature vectors that is an array
def gabor_feature_vector(images, filters):
    feature_vectors = {}
    for key,value in images.items():
        feature = []
        for img in value: 
            means = process(img, filters)
            
            minimum = min(means)
            maximum = max(means)
            for score in range(len(means)):
                means[score] = (means[score] - minimum) / (maximum - minimum)
            feature.append(means)
            
        feature_vectors[key] = feature
    return feature_vectors
            
# Makes convolution and take its mean. 
# Takes one image and all filters as parameters.
# Returns the mean that is feature vector
def process(img, filters):
    means = []
    for flt in filters:
        filtered_image = ndimage.convolve(img, flt)
        mean = np.mean(filtered_image)
        means.append(mean)
    return means

# takes all images and convert them to grayscale. 
# return a dictionary that holds all images category by category. 
def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        category = []
        path = folder + "/" + filename
        for cat in os.listdir(path):
            img = cv2.imread(path + "/" + cat,0)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img is not None:
                category.append(img)
        images[filename] = category
    return images

# This function produces 40 differenet gabor filters. It takes no parameter.
# Returns the filters that is a array holds each filter.
def build_filters():
    count = 0
    filters = []
    for theta in range(90):
        kern = cv2.getGaborKernel((3, 3), 8.0, count, 13.0, 0.9, 0, ktype=cv2.CV_32F)
        count += 2
        filters.append(kern)
    return filters

def main():
    
    images = load_images_from_folder('dataset/train')  # take all images category by category 
    test = load_images_from_folder("dataset/query") # take test images 
    
    
    ## gabor filter ##
    
    filters = build_filters() # take the all filters
    feature_vectors = gabor_feature_vector(images, filters) # create feature vectors for train dataset 
    test_vectors = gabor_feature_vector(test,filters) #create feature vectors for test dataset
    results = knn(feature_vectors, test_vectors) # call the knn function
    accuracy(results)  # calculates the accuracies and write the results to the console.
    images_print(images, feature_vectors, test_vectors, test)
    
    ## gabor filter end ##
    
    ## SIFT filter ##
    
    #sift_vectors = sift_filters(images) # extracts the sift vector fetaures for all train images
    #test_sift_vectors = sift_filters(test) # extracts the sift vector fetaures for all test images
    #results_sift = knn(sift_vectors, test_sift_vectors) # call the knn function
    #accuracy(results_sift) # calculates the accuracies and write the results to the console.
    #images_print(images, sift_vectors, test_sift_vectors, test)
    
    ## SIFT filter end ##
    
    ## Bag of Visual Words without spatial tiling ##
    
    
    #sifts = sift_features(images) 
    #descriptor_list = sifts[0] # Takes the descriptor list which is unordered one
    #all_bovw_feature = sifts[1] # Takes the sift features that is seperated class by class for train data
    #visual_words = kmeans(150, descriptor_list) # Takes the central points which is visual words
    #bovw_train = image_class(all_bovw_feature, visual_words) # Creates histograms for train data
    #test_bovw_feature = sift_features(test)[1] # Takes the sift features that is seperated class by class for test data
    #bovw_test = image_class(test_bovw_feature, visual_words) # Creates histograms for test data
    #results_bowl = knn(bovw_train, bovw_test) # Call the knn function
    #accuracy(results_bowl) # Calculates the accuracies and write the results to the console.
    #images_print(images, bovw_train, bovw_test, test)
    
    ## Bag of Visual Words End ##
    
    ## Bag of Visual Words with spatial tiling ##
    
    #images_tiling = tiling(images,500)
    #test_tile = tiling(test, 500)
    #sifts_tile = sift_features_tiling(images_tiling)
    #descriptor_list_tile = sifts_tile[0]
    #all_bovw_feature_tile = sifts_tile[1]
    #visual_words = kmeans(150, descriptor_list_tile)  
    #bovw_train = image_class_tiling(all_bovw_feature_tile, visual_words)
    #test_bovw_feature = sift_features_tiling(test_tile)[1]
    #bovw_test = image_class_tiling(test_bovw_feature, visual_words)
    #results_bowl = knn(bovw_train, bovw_test)
    #accuracy(results_bowl)
    #images_print(images, bovw_train, bovw_test, test)
    
    ## Bag of Visual Words End with spatial tiling End ##
main()

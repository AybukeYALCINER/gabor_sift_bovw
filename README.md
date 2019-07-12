## You can access the Dataset: https://drive.google.com/open?id=1PmeoRg3tbC_Cl94CYJ-1-gADmu77fYN0

# How to run code

To compare the actual labels and predicted labels for query images, I separate the images class by class like in train dataset.
To run the code in spyder or pycharm, just put the dataset folder in the same location with python file. Then run the code. 
<br>
# The organization of the code

After I wrote all functions, I call all of them in the main function. To run the code, I just call the main() function. All
other functions are called in main(). And I take the function in comment line, for example to see the accuracies for gabor filters
just comment out the gabor filter functions and comment in the others. 

# Details of implementation

NOT: I prefer to use dictionaries to separate class by class. All dictionaries are separated class by class.<br>
images = load_images_from_folder('dataset/train')  # take all images category by category <br>
test = load_images_from_folder("dataset/query") # take test images <br>

In the above code, just load the query images and train images into the dictionaries that are separated class by class
def load_images_from_folder(folder) => Takes the path of the folders as parameter and returns a dictionary.

## Gabor filter bank

To see the effect of gabor filter bank; call the below functions in the main function respectively;

filters = build_filters() <br>
feature_vectors = gabor_feature_vector(images, filters) <br>
test_vectors = gabor_feature_vector(test,filters) <br>
results = knn(feature_vectors, test_vectors) <br>
accuracy(results)  <br>

The definition of the functions as follows:<br>

def build_filters() => Takes no parameter and returns the array of gabor filters with different orientations.<br>
def gabor_feature_vector(images, filters) => Takes train images dictionary and test images dictionary as parameter. And returns the dictionary of convolution between each filter and each image. To achieve these, it calls process() function in it.<br>
def process(img, filters) => Takes one image and array of gabor filters as parameters. Then takes the mean of the each convolved filtered_images then return the array of it.<br>
def knn(images, tests) => Takes feature vectors of train and test images that we get them using  gabor_feature_vector() function. It is a classifying algorithm and the k value is 1 in this scenario. Return an array that holds number of images in query folder, number of images that we correctly predict the label and a dictionary that holds the number of coreectly predicted image and number of total images to calculate the class based accuracies.<br>
def accuracy(results) => Takes an array that return from knn(). Calculates the average and class based accuracies.
<br>
## SIFT feature vectors

To see the effect of SIFT feature vectors, call the below functions in the main function respectively;<br>

sift_vectors = sift_filters(images) <br>
test_sift_vectors = sift_filters(test) <br>
results_sift = knn(sift_vectors, test_sift_vectors)<br>
accuracy(results_sift) <br>
	
The definition of the functions as follows:<br>
 
def sift_filters(images) => Takes the dictionary of the images whose sift feature vectors should be extracted. Return a dictionary that holds the feature vectors of each image class by class<br>
def knn(images, tests) => Takes feature vectors of train and test images that we get them using  sift_filter() function. It is a classifying algorithm and the k value is 1 in this scenario. Return an array that holds number of images in query folder, number of images that we correctly predict the label and a dictionary that holds the number of correctly predicted image and number of total images to calculate the class based accuracies.<br>
def accuracy(results) => Takes an array that return from knn(). Calculates the average and class based accuracies.<br>
	
	
## Bag of Visual Words Without Spatial Tiling

To see the effect of bag of visual words, call the below functions in the main function respectively;<br>

sifts = sift_features(images) <br>
descriptor_list = sifts[0] # Takes the descriptor list which is unordered one <br>
all_bovw_feature = sifts[1] # Takes the sift features that is separated class by class for train data <br>
visual_words = kmeans(150, descriptor_list) # Takes the central points which is visual words <br>
bovw_train = image_class(all_bovw_feature, visual_words) # Creates histograms for train data <br>
test_bovw_feature = sift_features(test)[1] # Takes the sift features that is separated class by class for test data <br>
bovw_test = image_class(test_bovw_feature, visual_words) # Creates histograms for test data <br> 
results_bowl = knn(bovw_train, bovw_test) # Call the knn function <br>
accuracy(results_bowl) 	
	<br>
The definition of the functions as follows:<br>

def sift_features(images) => Takes just image dictionary and return an array whose first index holds the array of descriptor lists and the second one holds the dictionary of the descriptors but this time separated class by class<br>
def kmeans(k, descriptor_list) => Takes number of cluster and descriptor list as parameters and returns the central point of the clusters. This is just a clustering algorithm.<br>
def image_class(all_bovw, centers) => Takes the sift feature dictionary as parameter and the central points of the clusters that are visual words in this case. And returns a dictionary that holds histograms of the visual words for each image. To do that, it calls find_index() function.  <br>
def find_index(image, center) => Find the index of the closest central point to the each sift descriptor. Takes one of the sift vector and visual words as parameter and returns the index of the closest visual word.<br>
def knn(images, tests) => Takes the histograms of train and test images that we get them using  image_class() function. It is a classifying algorithm and the k value is 1 in this scenario. Return an array that holds number of images in query folder, number of images that we correctly predict the label and a dictionary that holds the number of correctly predicted image and number of total images to calculate the class based accuracies.<br>
def accuracy(results) => Takes an array that return from knn(). Calculates the average and class based accuracies.<br>
	
## Bag of Visual Words With Spatial Tiling

To see the effect of bag of visual words with spatial tiling, call the below functions in the main function respectively;<br>

images_tiling = tiling(images,500)<br>
test_tile = tiling(test, 500)<br>
sifts_tile = sift_features_tiling(images_tiling)<br>
descriptor_list_tile = sifts_tile[0]<br>
all_bovw_feature_tile = sifts_tile[1]<br>
visual_words = kmeans(200, descriptor_list_tile)  <br>
bovw_train = image_class_tiling(all_bovw_feature_tile, visual_words)<br>
test_bovw_feature = sift_features_tiling(test_tile)[1]<br>
bovw_test = image_class_tiling(test_bovw_feature, visual_words)<br>
results_bowl = knn(bovw_train, bovw_test)<br>
accuracy(results_bowl)<br>

The definition of the functions as follows:<br>

def tiling(images, k) => Takes images dictionary and number of tiles as parameters. Then split each images into number of tiles then a dictionary that holds the tiles of the images which are separated class by class. In this function find_multiplier(num) is called. <br>
def find_multiplier(num) => Helps the tiling function to find out one of the multipliers of the number of tile. Takes number of tile as parameter. And returns one of the multipliers. <br>
def sift_features_tiling(images) => Creates descriptors using sift library for each tile. Takes one parameter that is images dictionary that holds the tiles not the pictures itselves.Return an array whose first index holds the decriptor_list without an order. And the second index holds the sift_vectors dictionary which holds the descriptors but this is separated class by class.<br>
def kmeans(k, descriptor_list) => Takes number of cluster and descriptor list as parameters and returns the central point of the clusters. This is just a clustering algorithm.<br>
def image_class_tiling(all_bovw, centers) => Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class. And the second parameter is an array that holds the central points (visual words) of the k means clustering. Returns a dictionary that holds the histograms for each tiles of the images that are separated class by class. <br>
def knn(images, tests) => Takes the histograms of train and test images that we get them using  image_class() function. It is a classifying algorithm and the k value is 1 in this scenario. Return an array that holds number of images in query folder, number of images that we correctly predict the label and a dictionary that holds the number of correctly predicted image and number of total images to calculate the class based accuracies.<br>
def accuracy(results) => Takes an array that return from knn(). Calculates the average and class based accuracies.<br>

## Find the 5 most similar images from train dataset to the 3 images from 3 different class of query dataset.

images_print(images, feature_vectors, test_vectors, test) #for gabor filter bank<br>
images_print(images, sift_vectors, test_sift_vectors, test) #for sift feature vector<br>
images_print(images, bovw_train, bovw_test, test) #for bag of visual words with/without spatial tiling<br>

The definition of the images_print() as follows:<br>

def images_print(images, feature_vectors, test_vectors, test) => Takes image dictionary, 2 dictionaries that we pass to knn() and test dictionary as parameters respectively. Makes the concatenation of the 5 images and save them for each 3 different test image. Call closest() function. <br>
def closests(images, test) => Returns the 5 most similar categories and their index. Takes images dictionary and just one test image as parameters. 
<br>
## BONUS : I used L1-Distance(Manhattan Distance) 

def L1_dist(vec1, vec2) => Takes 2 vectors and return the distance between them






















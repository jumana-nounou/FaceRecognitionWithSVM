# FaceRecognitionWithSVM
Implementation of a face recognition system in python using Hog and SVM

The dataset used in this project is a collection of images of seven different individuals: Jumana, 
Farida, Maher, Khaled, Kroush , Ammar , Joana. The dataset was created with the team’s 
members and their friends. The images were then manually labeled with the corresponding 
individual's name.
The dataset contains a total of 23 images, with each individual having approximately 3-4 images. 
The images have varying resolutions, lighting conditions, and different angles making the task of 
face recognition challenging.

The first step in training the SVM model is to preprocess the dataset. We use the OpenCV library 
to detect faces in each image using a Haar Cascade Classifier. The detected face region is then 
resized to a target size of 388x388 pixels using the scikit-image library. Finally, the Histogram of 
Oriented Gradients (HOG) features are extracted from each image using the scikit-image library.

The HOG features and corresponding labels are used to train a linear SVM classifier using scikitlearn library. The dataset is split into 80% training and 20% testing data using the 
train_test_split function from scikit-learn. The HOG function was applied with number of 
orientation equal to 9, pixel per cell was 8x8, and the block norm was L2-Hys, and channel axis 
of two. The SVM was applied with a random state of value 42.

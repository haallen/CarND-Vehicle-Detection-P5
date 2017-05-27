"""
The goals / steps of this project are the following:

1. Perform a Histogram of Oriented Gradients (HOG) feature extraction on a 
    labeled training set of images and train a classifier Linear SVM classifier
2. Optionally, you can also apply a color transform and append binned color 
    features, as well as histograms of color, to your HOG feature vector.
Note: for those first two steps don't forget to normalize your features and 
    randomize a selection for training and testing.
3. Implement a sliding-window technique and use your trained classifier to 
    search for vehicles in images.
4. Run your pipeline on a video stream 
    (start with the test_video.mp4 and later implement on full project_video.mp4) 
    and create a heat map of recurring detections frame by frame to reject 
    outliers and follow detected vehicles.
5. Estimate a bounding box for vehicles detected.
"""
import numpy as np
import glob
import os
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lesson_functions import *
import pickle
#path to test data
dataPath = "/Users/hope/Documents/python/carND/CarND-Vehicle-Detection/data/"
classifierPath = "/Users/hope/Documents/python/carND/CarND-Vehicle-Detection/classifier.p"
#%% Step 1: Perform a Histogram of Oriented Gradients (HOG) feature extraction on
# a labeled training set of images and train a classifier Linear SVM classifier

#%% Step 2: Optionally, you can also apply a color transform and append binned 
    #color features, as well as histograms of color, to your HOG feature vector.

#%% Main Pipeline for training the classifier

carDataPath = dataPath + 'vehicles/'

cars = []
for d in os.walk(carDataPath):
    cars.extend(glob.glob(d[0] + '/*.png'))

notcarDataPath = dataPath + 'non-vehicles/'
notcars = []
for d in os.walk(notcarDataPath):
    notcars.extend(glob.glob(d[0] + '/*.png'))
    
vis = False
"""
vis = True
cars = cars[0:1]
notcars=notcars[0:1]
"""
t=time.time()
car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, visFlag=vis)
notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat, visFlag=vis)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

# Use a linear SVC 
svc = LinearSVC()

# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

# Check the prediction time for a single sample
t=time.time()

n_predict = len(y_test)
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

pDict = {}
pDict['svc']=svc
pDict['X_scaler'] = X_scaler
pickle.dump(pDict, open(classifierPath, "wb" ))


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/HOGviz.png
[image3]: ./output_images/heatmap.png
[image4]: ./output_images/search_windows.png
[image5]: ./output_images/detects1.png
[image6]: ./output_images/detects2.png
[image7]: ./output_images/detects3.png
[video1]: ./output_images/project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is located in the extract_features, single_img_features, and get_hog_features methods of lesson_functions.py located at this [link](https://github.com/haallen/CarND-Vehicle-Detection-P5/blob/master/lesson_functions.py) . I isolated this code because it is common to both the training and testing portions of this project. 

In my training [file](https://github.com/haallen/CarND-Vehicle-Detection-P5/blob/master/VehicleDetection_trainPipeline.py), I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of a `vehicle` and `non-vehicle` image:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Interestingly, HLS color space provided good accuracy values but resulted in inconsistent detections of vehicles.

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of orientations, pixels_per_cell, cells_per_block and colorspaces. Colorspace had the most effect on the accuracy of my classifier. I decided to go with HLS. Orientation had some effect while pixels_per_cell and cells_per_block did not appear to have a lot of effect on the overall accuracy

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I used the following features to train my classifier: binned color features, color histogram features, and HOG features. I found that the accuracy of my classifier increased when I used all 3. 

Once I calculated the feature data, I normalized the data, shuffled the data, and then split the data into training and test sets. I then fed this into a linear SVM classifier.

My total accuracy on the test data was 99.4%

I saved my trained classifier to a pickle file to be used later.

See lines 65-125 in my  [training code](https://github.com/haallen/CarND-Vehicle-Detection-P5/blob/master/VehicleDetection_trainPipeline.py)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for searching an image is in the search_window method of my [testing file](https://github.com/haallen/CarND-Vehicle-Detection-P5/blob/master/VehicleDetection_testPipeline.py).

The code for calling this method is in lines 251-292 of the same file. I created windows of 4 different scales to search. I also used the heatmap technique discussed in lectures to combine multiple detections and reduce false alarms. This is in lines 118-177.

Here's an example of the heatmap:

![alt text][image3]

Here's an example of my search windows. The red boxes are all the windows that were searched. The green boxes indicate that the image within the box was classified as a vehicle. The blue box is the result of the heatmap processing. 
![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In addition to the heatmap, window choice, and colorspace optimizations already mentioned I also optimized on window overlap. I found that overlapping some windows by more than 50% resulted in a better outcome. I balanced the benefits of overlap with the processing time to do the calculation.

Here's some examples of final results
![alt text][image5]
![alt text][image6]
![alt text][image7]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As mentioned earlier, I applied the heatmap algorithm and code presented in the video lectures to reduce false positives and combine bounding boxes. I provided examples of them above. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First and foremost, my approach is too computationally expensive. I looked into HOG subsampling and couldn't get it working, but think I know how. There also seem to be some accuracy issues with my multi-scale windows, so this subsampling technique would help with that.

I would also try to track the boxes between frames and perform some sort of smoothing to get a stable track. 

Sigh, and I would like to have it so the box goes around the entire car. I was just happy that it found the car at all without any false alarms. 

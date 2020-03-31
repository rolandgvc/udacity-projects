# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

FP.1  
I am storing the counts in a map where keys are the indexes of bounding boxes from the previous frame and values are maps where keys are the indexes of bounding boxes in the current frame and values are counts. Then, for each cv::DMatch object I look to find which bounding boxes each of the ends of the match belong to and update respective count if both ends were matched with the boxes. After that for each bounding box from the previous frame the box from the current frame with the highest count is selected.

FP.2  
I am using the median value of the distances to mitigate the problem with potential outliers instead of using the closest point (ee also FP.5).

FP.3  
I am computing the mean and standard deviation of the distribution of distances between matched points. Then, I am filtering out all points that are more than one s.d. away from the mean.

FP.4  
I am using median value for distance ratios to mitigate the outliers problem.

FP.5  
The general trend for TTC seems to be aligned with the distance to the car with both of them getting smaller with time. However, in frames 3-4 and 4-5 a reverse trend exists where the distance to the car decreases with the increase in TTC. Opon a visual examination, I noticed that the enclosed lidar points have a wider formation compared to adjacent, where the clusters seem tighter. The median value of distance to the point cloud is being altered. One possible way of avoiding outliers is filtering the point cloud using mean and standard deviation (as in FP.3) and calculating the distance to the closest point for use in TTC calculations.

Moreover, I've noticed a couple of frames where the certain detector-descriptor filter combinations deviate the camera-based TTC from Lidar TTC considerably. Specifically, FAST FREAK tends to deviate considerably from the LIDAR ground truth in the early frames (2-6), with values as high as 5.7 (see .xlsx). After frame 7, the camera TTC values catch up with the sensor.


FP.6  
The recorded data is in the xlsx file. There is a noticible error for the SIFT detector, probably caused by false matches from vehicles further down the road. A second plot has been added, with key points from region of interest only. We ca visualize the downwords trend TTC as we progress through frames.

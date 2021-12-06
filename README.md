# Face_mask_detection
Face Mask Detection with Machine Learning

## Introduction to Face Mask Detection
Face mask detection has a range of applications from capturing the movement of the face to facial recognition which at first requires the face to be detected with very good precision. Face detection is more relevant today as it is not only used on images, but also in video applications like real-time surveillance and face detection in videos.
![144886002-51275767-d499-4357-8b70-11a483c748fb](https://user-images.githubusercontent.com/95492893/144887668-3b65becf-23b5-42fb-8a1d-817bd2701589.png)![144886168-e9f6d79a-603a-4bbd-8964-793d2f641bd0](https://user-images.githubusercontent.com/95492893/144887678-de2e133e-060b-4177-953e-b45d8ceb3805.png)


High precision image classification is now possible with advances in convolutional networks. Pixel level information is often needed after face detection, which most face detection methods do not provide.

Obtaining pixel-level detail has been a difficult part of semantic segmentation. Semantic segmentation is the process of assigning a label to each pixel in the image.

## Process of Face Mask Detection with Machine Learning

Step 1: Extract face data for training.


Step 2: Train the classifier to classify faces in mask or labels without a mask.


Step 3: Detect faces while testing data using SSD face detector.


Step 4: Using the trained classifier, classify the detected faces.


In the third step of the above process, you have to think about what is the SSD face detector? Well, the SSD is a Single Shot Multibox Detector. This is a technique used to detect objects in images using a single deep neural network.

It is used for the detection of objects in an image. Using a basic architecture of the VGG-16 architecture, the SSD can outperform other object detectors such as YOLO and Faster R-CNN in terms of speed and accuracy.


## Face Mask Detection with Machine Learning

Now, let’s get started with the task of Face Mask Detection with Machine Learning by using the Python programming language. I will start this task by importing the necessary Python libraries that we need for this task:

![m1](https://user-images.githubusercontent.com/95492893/144885348-90597729-e0b4-4268-9547-88eafc0f419d.PNG)

## Creating Helper Functions
I will start this task by creating two helper functions:

![m2](https://user-images.githubusercontent.com/95492893/144885494-0537f33a-365e-4a8c-8b7d-d08df8fa5f18.PNG)

1. The getJSON function retrieves the json file containing the bounding box data in the training dataset.


2. The adjust_gamma function is a non-linear operation used to encode and decode luminance or tristimulus values in video or still image systems. Simply put, it is used to instil a little bit of light into the image. If gamma <1, the image will shift to the darker end of the spectrum and when gamma> 1, there will be more light in the image.

## Data Processing
The next step is now to explore the JSON data provided for the training:

![m3](https://user-images.githubusercontent.com/95492893/144885737-dbf30373-9d1e-47f1-af96-49d28fdbd4f8.PNG)
![m4](https://user-images.githubusercontent.com/95492893/144886002-51275767-d499-4357-8b70-11a483c748fb.PNG)

The Annotations field contains the data of all the faces present in a particular image.

There are different class names, but the real class names are face_with_mask and face_no_mask.

![m5](https://user-images.githubusercontent.com/95492893/144886168-e9f6d79a-603a-4bbd-8964-793d2f641bd0.PNG)

Using the mask and the non_mask labels, the bounding box data of the json files is extracted. The faces of a particular image are extracted and stored in the data list with its tag for the learning process.

![m6](https://user-images.githubusercontent.com/95492893/144886350-c09c0034-e462-4195-af6c-e3d3dfc28dd8.PNG)
![m7](https://user-images.githubusercontent.com/95492893/144886492-d379198d-0165-4bc4-8fe9-3f8049fa6b7f.PNG)

![image](https://user-images.githubusercontent.com/95492893/144886535-30d68cba-e645-406e-9434-e870788954fc.png)

The visualization above tells us that the number of mask images> Number of images without a mask, so this is an unbalanced dataset. But since we’re using a pre-trained SSD model, which is trained to detect unmasked faces, this imbalance wouldn’t matter much.

But let’s reshape the data before training a neural network:
![m8](https://user-images.githubusercontent.com/95492893/144886671-350c82cd-941d-4c32-9cd9-9051f27ad19c.PNG)

## Training Neural Network for Face Mask Detection
Now the next step is to train a Neural Network for the task of Face Mask Detection with Machine Learning:

![m9](https://user-images.githubusercontent.com/95492893/144886931-24482dbf-48b6-410e-bde7-b0669e9223b8.PNG)
![m10](https://user-images.githubusercontent.com/95492893/144886964-7ee499e0-a9fa-4dba-832e-e1a1dabd225e.PNG)

## Testing The Model
The test dataset contains 1698 images and to evaluate the model so I took a handful of images from this dataset as there are no face tags in the dataset:

![m11](https://user-images.githubusercontent.com/95492893/144887337-7b995a7c-52db-4221-8b1e-bd3b82819f0c.PNG)
![m12](https://user-images.githubusercontent.com/95492893/144887386-ec49a768-f0ba-4219-a4a4-37460a5a74c8.PNG)

![image](https://user-images.githubusercontent.com/95492893/144887483-686af922-aca3-4b6d-b4bf-3c94c0f71c50.png)


By analyzing the output above, we can observe that the whole system works well for faces that have spatial dominance. But fails in the case of images where the faces are small and take up less space in the overall image.

For best results, different image preprocessing techniques can be used, or the confidence threshold can be kept lower, or one can try different blob sizes.























# Buzzard Detector

## Feature Extraction
1. Extract SURF/SIFT/ORB feature descriptors
2. Build a "visual vocabulary" according to Bag of Words(BoW) model
3. Quantize each image into a histogram vector based on the vocabulary

## Recognition
1. Linear SVM
2. Train the SVM on those histogram vectors from images in training set


## Localization
1. Extract descriptors from several template images
2. FLANN matching for descriptors from template images to the novel image
3. Filter out some outliers from remaining matches(cyan circles) utilizing median and std. of those matches
4. Use the final remaining key points(red circles) to build up the bounding box
5. Maintain a queue storing a sequence of bbox locations from the near history in order to smooth the variation of bbox locations 

## Results

- Extracted keypoints
![Alt text](results/imgs/localization/SURF_im_video_4_1_kps.jpg?raw=true "key points")


- Localization
![Alt text](results/imgs/localization/SURF_im_video_4_1_bb.jpg?raw=true "bbox")


- [Video demo](https://youtu.be/50UbhD-VNU0)
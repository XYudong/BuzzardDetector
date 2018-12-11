dataset(index 0):
1. train_SIFT_negative_0.txt and
   train_SIFT_positive_0.txt are in shape (56,50)
each row is a histogram feature vector of an image.

2. test_SIFT_positive_0.txt in shape (14, 50)
   test_SIFT_negative_0.txt in shape (7, 50)

3. ORB: row vector is in length 30. other dimensions should be the same as above.


11/21/2018:
--the first data set
    added SURF features  

--the second data set(videos) 
ORB, SIFT and SURF:
Train:
    positive: 527
    negative: 280
    
Test:
    positive: 81
    negative: 28
    
foler 0:
    Containing BoW feature vectors of the first data set for three different feature types;
    
folder 1:
    Containing BoW feature vectors of the second data set(i.e. images extracted from videos);
    
    
12/6/2018:
Update the second dataset(index 1):
ORB, SIFT and SURF:
Train:
    positive: 563
    negative: 563
    
Test:
    positive: 89
    negative: 87

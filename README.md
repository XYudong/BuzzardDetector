# Buzzard Recognizer

## Feature Extraction
1. Extract SIFT/ORB/SURF feature descriptors
2. Build a "vocabulary" according to Bag of Word model
3. Quantize each image into a histogram vector based on the vocabulary

## Recognition
1. Linear SVM
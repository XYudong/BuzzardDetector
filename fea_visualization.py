import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import cv2
import time
import os


img = cv2.imread('output/templates/imgs/pos_002.jpg', 0)     # test example
if any(np.array(img.shape) > 1000):
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
img = cv2.GaussianBlur(img, (3, 3), sigmaX=0)
print('img shape: ' + str(img.shape))

# SIFT
t1 = time.time()
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.07)
kp, des = sift.detectAndCompute(img, None)

t2 = time.time()
print("\nSIFT time: " + str(t2 - t1))

img_sift = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0))
# img_sift = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(len(kp))
print(des.shape)

#####

# ORB
t1 = time.time()
orb = cv2.ORB_create(nfeatures=250, scaleFactor=1.2)
kp_orb, des_orb = orb.detectAndCompute(img, None)

t2 = time.time()
print("\nORB time: " + str(t2 - t1))

img_orb = cv2.drawKeypoints(img, kp_orb, img, color=(0, 255, 0))

print(len(kp_orb))
print(des_orb.shape)

#######

# SURF
t1 = time.time()
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=900, extended=True)
kp_surf, des_surf = surf.detectAndCompute(img, None)

t2 = time.time()
print("\nSURF time: " + str(t2 - t1))
print(len(kp_surf))
print(des_surf.shape)

img_surf = cv2.drawKeypoints(img, kp_surf, img, color=(0, 255, 0))

f1 = plt.figure(1)
plt.imshow(img_sift, 'gray')
plt.title('SIFT')

f2 = plt.figure(2)
plt.imshow(img_orb, 'gray')
plt.title('ORB')

f3 = plt.figure(3)
plt.imshow(img_surf, 'gray')
plt.title('SURF')

plt.show()


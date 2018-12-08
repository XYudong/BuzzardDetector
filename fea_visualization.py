import numpy as np
from sklearn.utils import resample
import matplotlib.pyplot as plt
import cv2
import time
import os


def compress_img():
    """compress all images in the path"""
    in_path = 'data/train/positive/00/'
    out_path = 'data/train/positive/0/'
    names = os.listdir(in_path)
    for i, name in enumerate(names):
        img = cv2.imread(in_path + name, 0)
        img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out_path + name, img)

    return


img = cv2.imread('data/train/positive/1/im_video2_4_41.jpg', 0)     # test example
img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
print('img shape: ' + str(img.shape))

# SIFT
t1 = time.time()
sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.075)
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
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=950, extended=True)
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


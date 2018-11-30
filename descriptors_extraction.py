import cv2
import numpy as np
from sklearn.utils import resample
from scipy.cluster.vq import vq, whiten
import time
import os


def get_des(typeIn, imgIn):
    """find key points, return descriptors according to @type and a descriptor image"""
    t1 = time.time()
    if typeIn == "SIFT":
        sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.06)
        kp, des = sift.detectAndCompute(imgIn, None)
    elif typeIn == "ORB":
        orb = cv2.ORB_create(nfeatures=600, scaleFactor=1.2)
        kp, des = orb.detectAndCompute(imgIn, None)
    elif typeIn == "SURF":
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=900, extended=True)
        kp, des = surf.detectAndCompute(imgIn, None)
    else:
        print("invalid feature type")
        return

    t2 = time.time()
    print(typeIn + " time: " + str(t2 - t1))
    img_fea = cv2.drawKeypoints(imgIn, kp, imgIn, color=(0, 255, 0))
    return img_fea, des


def preprocess_img(img_path):
    if isinstance(img_path, str):
        img_in = cv2.imread(img_path, 0)
    else:
        img_in = img_path       # means the input is already the image needed

    if any(np.array(img_in.shape) > 1000):
        img_in = cv2.resize(img_in, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    img_in = cv2.GaussianBlur(img_in, (3, 3), sigmaX=0)
    return img_in


def get_all_des(typeIn, path):
    """prepare a corpus for the vocabulary"""
    names = os.listdir(path)
    descriptors = np.array([])
    for name in names:
        img_tr = preprocess_img(path + name)
        img_des, des = get_des(typeIn, img_tr)
        if descriptors.size == 0:
            descriptors = des
        else:
            descriptors = np.vstack((descriptors, des))

    if len(descriptors) > 10000:
        descriptors_new = resample(descriptors, n_samples=10000, replace=False, random_state=666)
    else:
        descriptors_new = descriptors

    return descriptors_new


def quantize_des(des, vocIn, n_clu):
    """quantize all whitened descriptors of an image"""
    img_hist = np.zeros((1, n_clu))[0]
    words, dist = vq(des, vocIn)
    for w in words:
        img_hist[w] += 1

    # normalization
    img_hist = img_hist / len(words)
    # return an array
    return img_hist


def imgs_quantization(in_path, in_type, in_voc, n_clus):
    """
    Main workflow here
    quantize each image in the path as a vector and stack them vertically
    """
    names = os.listdir(in_path)
    img_hists = np.array([])
    for name in names:
        img_new = preprocess_img(in_path + name)
        img_annotated, des = get_des(in_type, img_new)
        des_wh = whiten(des)
        img_hist_vec = quantize_des(des_wh, in_voc, n_clus)
        if img_hists.size == 0:
            img_hists = img_hist_vec
        else:
            img_hists = np.vstack((img_hists, img_hist_vec))
    return img_hists


def img_quantizer(img, fea_type, voc, n_clus):
    img_new = preprocess_img(img)
    img_annotated, des = get_des(fea_type, img_new)
    des_wh = whiten(des)
    img_hist_vec = quantize_des(des_wh, voc, n_clus)

    return img_hist_vec

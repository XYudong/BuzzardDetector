from descriptors_extraction import get_des
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os
import pickle
from copy import deepcopy


def aug_imgs(in_path, out_path):
    """augment images from in_path"""
    names = os.listdir(in_path)
    aug = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.01*255), iaa.Fliplr(p=0.5), iaa.Affine(shear=-10)],
                         random_order=True)
    for i, name in enumerate(names):
        img = cv2.imread(in_path + name, 0)
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        img_new = aug.augment_image(img)
        cv2.imwrite(out_path + "videoAug_neg_2_"+str(i)+".jpg", img_new)

    return


def extract_template(temp_dir, fea_type):
    """extract features from provided images to form a template"""
    kps = []
    descriptors = np.array([])
    in_path = temp_dir + 'imgs/'        # images
    names = os.listdir(in_path)
    for i, name in enumerate(names):
        img = cv2.imread(in_path + name, 0)
        kp, des = get_des(fea_type, img)
        if descriptors.size == 0:
            kps = kp
            descriptors = des
        else:
            kps.extend(kp)
            descriptors = np.vstack((descriptors, des))

    with open(temp_dir + fea_type + '_template_0.pickle', 'wb') as ff:
        pickle.dump(descriptors, ff)

    # with open(temp_dir + fea_type + '_template_0.pickle', 'rb') as f:
    #     template = pickle.load(f)

    return

# # augment existing images
# path_to_img = "data/train/negative/2/"
# output_path = "data/train/negative/2_aug/"
# aug_imgs(path_to_img, output_path)


# extract features from template images
temp_dir = 'output/templates/'      # templates directory
extract_template(temp_dir, 'ORB')





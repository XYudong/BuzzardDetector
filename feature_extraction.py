from descriptors_extraction import imgs_quantization
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os


def aug_imgs(in_path, out_path):
    names = os.listdir(in_path)
    aug = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0.04*255), iaa.Fliplr(1), iaa.Affine(shear=10)],
                         random_order=True)
    for i, name in enumerate(names):
        img = cv2.imread(in_path + name, 0)
        img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
        img_new = aug.augment_image(img)
        cv2.imwrite(out_path + "aug_pos_2_"+str(i)+".jpg", img_new)

    return


def extract_fea_vec(dataset, fea_type):
    """extract histogram feature vector in fea_type from dataset with a pre-trained vocabulary"""
    # setting
    voc_path = "voc_output/"

    if fea_type == "ORB":
        voc_name = "myVoc_" + fea_type + "_0.txt"
        k = 30
    elif fea_type == "SIFT" or fea_type == "SURF":
        voc_name = "myVoc_" + fea_type + "_0.txt"
        k = 50
    else:
        print("invalid fea_type\n")
        return

    # load pre-built visual word vocabulary
    voc = np.loadtxt(voc_path + voc_name)

    # calculate a histogram feature vector for each input image from path provided
    print('processing new image(s)\n')
    img_hist_vecs_pos = imgs_quantization(positive_path, fea_type, voc, k)
    img_hist_vecs_neg = imgs_quantization(negative_path, fea_type, voc, k)

    print('shape positive samples: ' + str(img_hist_vecs_pos.shape) + '\n')
    print('shape negative samples: ' + str(img_hist_vecs_neg.shape) + '\n')

    np.savetxt(dataset + "_" + fea_type + "_positive_1.txt", img_hist_vecs_pos)
    np.savetxt(dataset + "_" + fea_type + "_negative_1.txt", img_hist_vecs_neg)

    return


# call functions to augment existing images
# path_to_img = "data/train/positive/"
# output_path = "data/video/image/positive_aug/"
# aug_imgs(path_to_img, output_path)


# setting
dataset = "test"
fea_type = "SURF"

# input images path
positive_path = "data/" + dataset + "/positive/1/"
negative_path = "data/" + dataset + "/negative/1/"

extract_fea_vec(dataset, fea_type)








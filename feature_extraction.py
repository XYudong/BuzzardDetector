from descriptors_extraction import imgs_quantization
import cv2
import numpy as np


def extract_fea_vec(dataset, fea_type, pos_path, neg_path):
    """extract histogram feature vector in fea_type for each image from dataset with a pre-trained vocabulary"""
    # setting
    voc_path = "voc_output/"

    if fea_type == "ORB":
        k = 30
    elif fea_type == "SIFT" or fea_type == "SURF":
        k = 50
    else:
        print("invalid fea_type\n")
        return

    # load pre-built visual word vocabulary
    voc_name = "myVoc_" + fea_type + "_01.txt"
    voc = np.loadtxt(voc_path + voc_name)

    # calculate a histogram feature vector for each input image from path provided
    print('processing new image(s)\n')
    img_hist_vecs_pos = imgs_quantization(pos_path, fea_type, voc, k)
    img_hist_vecs_neg = imgs_quantization(neg_path, fea_type, voc, k)

    print('shape positive samples: ' + str(img_hist_vecs_pos.shape) + '\n')
    print('shape negative samples: ' + str(img_hist_vecs_neg.shape) + '\n')

    np.savetxt("output/" + dataset + "_" + fea_type + "_positive01_1.txt", img_hist_vecs_pos)
    np.savetxt("output/" + dataset + "_" + fea_type + "_negative01_1.txt", img_hist_vecs_neg)

    return


# setting
dataset = "train"
fea_type = "ORB"

# input images path
positive_path = "data/" + dataset + "/positive/1/"
negative_path = "data/" + dataset + "/negative/1/"

extract_fea_vec(dataset, fea_type, positive_path, negative_path)








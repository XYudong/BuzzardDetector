from descriptors_extraction import get_des
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os
import pickle


class PointQueue:
    """
    Each element is a point with coordinates (x, y)
    """
    def __init__(self, maxsize=10):
        self.items = []
        self.max_size = maxsize

    def isEmpty(self):
        return self.items == []

    def isFull(self):
        return self.size() == self.max_size

    def push(self, item):
        if not self.isFull():
            item = self._smooth(item)
            self.items.insert(0, item)
        else:
            print("full queue\n")

    def remove(self):
        if not self.isEmpty():
            self.items.pop()
        else:
            print("empty queue\n")

    def size(self):
        return len(self.items)

    def mean(self):
        arr = np.array(self.items)
        return tuple(int(ele) for ele in np.mean(arr, 0))

    def get(self, idx):
        if -self.size() <= idx < self.size():
            return self.items[idx]
        else:
            print("index out of scope")

    def _smooth(self, item):
        if not self.isEmpty():
            item = list(item)
            gradient = 30       # maximum changes on each coordinate
            last_item = self.items[-1]
            x_diff = item[0] - last_item[0]
            y_diff = item[1] - last_item[1]
            item[0] = item[0] if abs(x_diff) < gradient else last_item[0] + abs(x_diff) / x_diff * gradient
            item[1] = item[1] if abs(y_diff) < gradient else last_item[1] + abs(y_diff) / y_diff * gradient

        return tuple(item)

    def clean(self):
        self.items = []
        return


def compress_img():
    """compress all images in the path"""
    in_path = 'output/templates/rgb/'
    out_path = 'output/templates/imgs/'
    names = os.listdir(in_path)
    for i, name in enumerate(names):
        img = cv2.imread(in_path + name, 0)
        if any(np.array(img.shape) > 1000):
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(out_path + name, img)

    return


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
        if any(np.array(img.shape) > 1000):
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        print(img.shape)
        kp, des = get_des(fea_type, img)
        if descriptors.size == 0:
            kps = kp
            descriptors = des
        else:
            kps.extend(kp)
            descriptors = np.vstack((descriptors, des))

    print("template descriptors shape: " + str(descriptors.shape))
    with open(temp_dir + fea_type + '_template_0.pickle', 'wb') as ff:
        pickle.dump(descriptors, ff)

    # with open(temp_dir + fea_type + '_template_0.pickle', 'rb') as f:
    #     template = pickle.load(f)

    return

# # augment existing images
# path_to_img = "data/train/negative/2/"
# output_path = "data/train/negative/2_aug/"
# aug_imgs(path_to_img, output_path)


#  compress images
# compress_img()


# extract features from template images
temp_dir = 'output/templates/'      # templates directory
extract_template(temp_dir, 'SIFT')





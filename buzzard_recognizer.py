import cv2
import numpy as np
import pickle
import time

from descriptors_extraction import img_quantizer


def extract_fea_vec(img_in, fea_type):
    """extract histogram feature vector in fea_type from dataset with a pre-trained vocabulary"""
    if img_in is None:
        print('input image is None')
        return

    if fea_type == "ORB":
        voc_name = "myVoc_" + fea_type + "_0.txt"
        k = 30
    elif fea_type == "SIFT" or fea_type == "SURF":
        voc_name = "myVoc_" + fea_type + "_0.txt"
        k = 50
    else:
        print("invalid feature type\n")
        return

    # setting
    voc_path = "voc_output/"
    # load pre-built visual word vocabulary
    voc = np.loadtxt(voc_path + voc_name)

    # calculate a histogram feature vector for the input image
    # print('processing new image(s)\n')
    img_hist_vec = img_quantizer(img_in, fea_type, voc, k)
    img_hist_vec = img_hist_vec.reshape(1, -1)

    print('shape: ' + str(img_hist_vec.shape) + '\n')

    return img_hist_vec


def recognize_fea_vec(img_vector, fea_type):
    type_list = ['ORB', 'SIFT', 'SURF']
    if fea_type not in type_list:
        print('invalid feature type\n')
        return
    model_path = 'pre-trained models/'
    filename = model_path + fea_type + '_model.sav'
    clf = pickle.load(open(filename, 'rb'), encoding='latin1')

    output = clf.predict(img_vector)

    return output


def main(fromVideo=True, fea_type='SURF'):
    if fromVideo:
        cap = cv2.VideoCapture(0)
        while True:
            t1 = time.time()

            grabbed, frame = cap.read()
            if not grabbed:
                print('fail to open the video\n')
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            vec = extract_fea_vec(frame, fea_type)
            res = recognize_fea_vec(vec, fea_type)

            t2 = time.time()

            cv2.putText(frame, "res: " + str(res), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print('This takes: ' + str(t2-t1) + ' seconds\n')

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        im = cv2.imread('data/test/negative/1/im_video_1_20.jpg', 0)
        vec = extract_fea_vec(im, fea_type)
        res = recognize_fea_vec(vec, fea_type)
        print('result is ' + str(res))
    return


main()

import cv2
import numpy as np
import pickle
import time
from utilis import PointQueue
from imutils.video import FPS
from matplotlib import pyplot as plt


from descriptors_extraction import img_quantizer


def extract_fea_vec(img_in, fea_type):
    """extract histogram feature vector in fea_type from dataset with a pre-trained vocabulary"""
    if img_in is None:
        print('input image is None')
        return

    if fea_type == "ORB":
        voc_name = "myVoc_" + fea_type + "_01.txt"
        k = 30
    elif fea_type == "SIFT" or fea_type == "SURF":
        voc_name = "myVoc_" + fea_type + "_01.txt"
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
    kp, des, img_hist_vec = img_quantizer(img_in, fea_type, voc, k)
    img_hist_vec = img_hist_vec.reshape(1, -1)

    # print('vector shape: ' + str(img_hist_vec.shape) + '\n')

    return kp, des, img_hist_vec


def recognize_fea_vec(img_vector, fea_type):
    type_list = ['ORB', 'SIFT', 'SURF']
    if fea_type not in type_list:
        print('invalid feature type\n')
        return
    model_path = 'pre-trained_models/'
    filename = model_path + fea_type + '_model01.sav'
    clf = pickle.load(open(filename, 'rb'), encoding='latin1')

    output = clf.predict(img_vector)

    return output


def match_fea(in_kp, in_des, fea_type):
    """
    :param in_kp: key points
    :param in_des:  descriptors from train image
    :param fea_type:
    :return:
    """
    temp_dir = 'output/templates/'      # templates directory
    with open(temp_dir + fea_type + '_template_0.pickle', 'rb') as f:
        template_des = pickle.load(f)       # descriptors from query images

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(template_des, in_des, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    out_kp = [in_kp[mm.trainIdx] for mm in good]

    return out_kp


def filter_kps(in_kps):
    num = len(in_kps)
    xs = np.zeros(num)
    ys = np.zeros(num)
    for i, kp in enumerate(in_kps):
        pt = kp.pt
        xs[i] = pt[0]
        ys[i] = pt[1]

    x_std = np.std(xs)
    y_std = np.std(ys)
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    out_x = []
    out_y = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        if abs(x - x_mean) < 1.3*x_std and abs(y - y_mean) < 1.3*y_std:
            out_x.append(x)
            out_y.append(y)
    return out_x, out_y


def get_bbox(xs, ys):
    x_med = np.median(xs)
    y_med = np.median(ys)

    bbox = []
    width = np.max(xs) - np.min(xs)
    height = np.max(ys) - np.min(ys)
    # set a minimum for the bb
    width = width if width > 15 else 15
    height = height if height > 15 else 15

    # left_top: the corner with the lowest values in both x, y coordinates
    left_top = (int(x_med - 0.7*width), int(y_med - 0.7*height))
    right_bottom = (int(x_med + 0.7*width), int(y_med + 0.7*height))

    bbox.append(left_top)
    bbox.append(right_bottom)

    return bbox


def filter_bbox(bbox):
    """return mean coordinates of a queue of bbox from history"""
    if not LeftTop_queue.isFull():
        LeftTop_queue.push(bbox[0])
        RightBot_queue.push(bbox[1])
    else:
        LeftTop_queue.remove()
        RightBot_queue.remove()
        LeftTop_queue.push(bbox[0])
        RightBot_queue.push(bbox[1])

    bbox[0] = LeftTop_queue.mean()
    bbox[1] = RightBot_queue.mean()

    return bbox


def draw_bbox(im, kps):
    # kps: key points after kNN matching
    xs, ys = filter_kps(kps)
    print('find ' + str(len(xs)) + ' matches\n')

    bbox = get_bbox(xs, ys)
    bbox = filter_bbox(bbox)

    # draw matched key points
    im = cv2.drawKeypoints(im, kps, im, color=(100, 200, 0))
    # draw filtered key points
    for x, y in zip(xs, ys):
        cv2.circle(im, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=2)
    # draw bounding box
    cv2.rectangle(im, *bbox, (0, 225, 100), thickness=3)  # draw bounding box

    return im


def main(fromVideo=True, fea_type='SIFT'):
    if fromVideo:
        path_to_video = "data/video/new/test_5.mp4"
        cap = cv2.VideoCapture(path_to_video)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('results/output_5.avi', fourcc, 20.0, (960, 540))

        global neg_times        # number of continuous negative recognition result
        fps = FPS().start()
        while cap.isOpened():
            # t1 = time.time()

            grabbed, frame = cap.read()
            if not grabbed:
                print('fail to open the video\n')
                break

            if any(np.array(frame.shape) > 1000):
                frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            kp, des, vec = extract_fea_vec(frame_gray, fea_type)
            res = recognize_fea_vec(vec, fea_type)

            say = "Woo, a Buzzard!" if res == 1 else "wait..."
            cv2.putText(frame, "res: " + str(say), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if res == 1:
                kps = match_fea(kp, des, fea_type)
                frame = draw_bbox(frame, kps)
                neg_times = 0
            else:
                neg_times += 1
                if neg_times > 3:
                    LeftTop_queue.clean()
                    RightBot_queue.clean()

            out.write(frame)

            # print("frame shape: " + str(frame.shape))
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            fps.update()
            # t2 = time.time()
            # print('This takes: ' + str(t2-t1) + ' seconds\n')

        fps.stop()
        print("approximate FPS: " + str(fps.fps()))

    else:
        im_name = 'im_video_4_1.jpg'
        im = cv2.imread('data/test/positive/1/' + im_name, 1)
        # im = cv2.imread('data/train/positive/0/pos_35.jpg', 1)

        if any(np.array(im.shape) > 1000):
            im = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        kp, des, vec = extract_fea_vec(im_gray, fea_type)
        res = recognize_fea_vec(vec, fea_type)
        # print('result is ' + str(res))

        say = "Woo, a Buzzard!" if res == 1 else "wait..."
        cv2.putText(im, "res: " + str(say), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 155, 200), 2)
        if res == 1:
            kps = match_fea(kp, des, fea_type)
            im = draw_bbox(im, kps)
            # im = cv2.drawKeypoints(im, kp, im, color=(0, 200, 0))

        out_path = 'results/imgs/'
        cv2.imwrite(out_path + fea_type + "_" + im_name, im)
        while True:
            cv2.imshow('Frame', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return


LeftTop_queue = PointQueue(4)      # FIFO queue for top_left point
RightBot_queue = PointQueue(4)
neg_times = 0
main(fromVideo=True, fea_type='SURF')


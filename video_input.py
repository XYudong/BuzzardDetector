import numpy as np
import cv2

"""extract and images from a video"""
video_idx = 9
path_to_video = "data/video/new/video2_" + str(video_idx) + ".mp4"
path_to_image = "data/video/image2/"

cap = cv2.VideoCapture(path_to_video)

i = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if frame is None:
        print("video finished")
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)

    # pick some frames and save them to image files
    num = 10
    remainder = 1
    if i % num == remainder:
        cv2.imwrite(path_to_image + "im_video2_" + str(video_idx) + "_" + str(int(i/(num+remainder))) + ".jpg", gray)

    i += 1

    # Display the resulting frame
    # cv2.imshow('frame', gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

print(i)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


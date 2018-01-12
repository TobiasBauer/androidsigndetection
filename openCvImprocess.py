from PIL import Image
import glob
import cv2
import numpy as np
import os

imgpath = "images/original/90/918.jpeg"
img = cv2.imread(imgpath, 1)

imgname = os.path.splitext(imgpath)[0]
imgextension = os.path.splitext(imgpath)[1]
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print(imgname)

lower_red = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
upper_red = cv2.inRange(hsv_image, (160, 100, 100), (179, 255, 255))

red_combined = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
red_combined = cv2.GaussianBlur(red_combined, (9, 9), 2, 2)

cv2.imshow("hogo", red_combined)
cv2.waitKey(0)
circles = cv2.HoughCircles(red_combined, cv2.HOUGH_GRADIENT, 1, (red_combined.shape[0]) / 2,
                           param1=100, param2=20, minRadius=10, maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))
    c = 0
    height, width, chan = img.shape
    for i in circles[0, :]:
        # draw the outer circle
        # cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        # cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
        if i is None or i[2] > width/2:
            continue
        #cropped = np.empty(i[2]*i[2], dtype='int64')


        print("x: " + str(i[0]) + "   y: " + str(i[1]) + "   rad: " + str(i[2]) + "  width: " + str(width) + " height: " + str(height))
        x1 = i[1] - i[2]
        x2 = i[1] + i[2]
        y1 = i[0] - i[2]
        y2 = i[0] + i[2]
        print("x1: " + str(x1) + "x2: " + str(x2) + "y1: " + str(y1) + "y2: " + str(y2))
        cropped = img[x1: x2, y1: y2]
        img2020 = cv2.resize(cropped, (20, 20))
        hsv_chans = cv2.split(img2020)
        # retval, binary_image = cv2.threshold(hsv_chans[2], 150, 255, cv2.THRESH_BINARY)

        cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2, 8, 0)
        cv2.imshow('binary' + str(c), hsv_chans[2])
        c += 1
        cv2.waitKey(0)

cv2.imshow('Popo', red_combined)
cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

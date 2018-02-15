from PIL import Image
import argparse
import cv2
import glob
import numpy as np
import os

if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="test.jpg", type=str,
                        help="Path to image")
    args = parser.parse_args()
    imgpath = args.image
img = cv2.imread(imgpath, 1)

imagename, imageextension = os.path.splitext(os.path.basename(imgpath))

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
upper_red = cv2.inRange(hsv_image, (160, 100, 100), (179, 255, 255))
red_combined = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
red_combined = cv2.GaussianBlur(red_combined, (9, 9), 2, 2)

cv2.waitKey(0)
circles = cv2.HoughCircles(red_combined, cv2.HOUGH_GRADIENT, 1, (red_combined.shape[0]) / 2,
                           param1=100, param2=20, minRadius=15, maxRadius=0)
if circles is not None:
    circles = np.uint16(np.around(circles))
    c = 0
    height, width, chan = img.shape

    for i in circles[0, :]:
        if i is None or i[2] > width/2:
            continue


        print("x: " + str(i[0]) + "   y: " + str(i[1]) + "   rad: " + str(i[2]) + "  width: " + str(width) + " height: " + str(height))
        x1 = i[1] - i[2]
        x2 = i[1] + i[2]
        y1 = i[0] - i[2]
        y2 = i[0] + i[2]
        print("x1: " + str(x1) + "x2: " + str(x2) + "y1: " + str(y1) + "y2: " + str(y2))
        cropped = img[x1: x2, y1: y2]
        img2020 = cv2.resize(cropped, (20, 20))
        hsv_chans = cv2.split(img2020)

        cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2, 8, 0)

        retval, binary_image = cv2.threshold(hsv_chans[2], 150, 255, cv2.THRESH_BINARY)

        croppedpath = imagename + "(" + str(c) + ")" + imageextension
        cv2.imwrite(croppedpath, binary_image)
        cv2.imshow("Binary", binary_image)
        img2020 = cv2.resize(binary_image, (20, 20))
        cv2.imwrite(croppedpath, binary_image)
        c += 1
        cv2.waitKey(0)

cv2.imshow('Popo', red_combined)
cv2.imshow('detected circles', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

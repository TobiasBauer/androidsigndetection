from PIL import Image
import glob
import cv2
import numpy as np
import os

for dirname in glob.glob('images/original/*'):
    print("dirname: " + dirname)
    if os.listdir(dirname) != []:
        for filename in glob.glob(dirname + "/*"):
            print(" filepath: " + filename)
            imgpath = filename
            speeddirectory = os.path.split(os.path.split(imgpath)[0])[1]
            print("speeddir: " + speeddirectory)

            img = cv2.imread(imgpath, 1)
            imgname = os.path.splitext(imgpath)[0]
            imgextension = os.path.splitext(imgpath)[1]

            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_red = cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
            upper_red = cv2.inRange(hsv_image, (160, 100, 100), (179, 255, 255))

            red_combined = cv2.addWeighted(lower_red, 1.0, upper_red, 1.0, 0.0)
            red_combined = cv2.GaussianBlur(red_combined, (9, 9), 2, 2)
            # red_combined = cv2.Canny(red_combined, 100, 200)

            circles = cv2.HoughCircles(red_combined, cv2.HOUGH_GRADIENT, 1, (red_combined.shape[0]) / 2,
                                       param1=100, param2=20, minRadius=15, maxRadius=0)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                j = 0
                for i in circles[0, :]:
                    # draw the outer circle
                    # cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                    # draw the center of the circle
                    # cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
                    if i is None:
                        continue
                    print("i0: " + str(i[0]) + "   i1: " + str(i[1]) + "     i2: " + str(i[2]))
                    cv2.rectangle(img, (i[0] - i[2], i[1] - i[2]), (i[0] + i[2], i[1] + i[2]), (0, 255, 0), 2, 8, 0)
                    cropped = img[i[1] - i[2]: i[1] + i[2], i[0] - i[2]: i[0] + i[2]]
                    img2020 = cv2.resize(cropped, (20, 20))
                    hsv_chans = cv2.split(img2020)
                    retval, binary_image = cv2.threshold(hsv_chans[2], 150, 255, cv2.THRESH_BINARY)
                    print(binary_image)
                    # cv2.imshow('cropped', cropped)
                    imagename, imageextension = os.path.splitext(os.path.basename(imgpath))
                    croppedpath = "images/cropped/" + speeddirectory + "/" + imagename + "(" + str(j) + ")" + imageextension
                    cv2.imwrite(croppedpath, binary_image)
                    print("Cropped path: " + croppedpath)
                    j += 1
                #cv2.imshow('Popo', red_combined)
                #cv2.imshow('detected circles', img)

    else:
        print(dirname + " Empty")
print("Conversion done")

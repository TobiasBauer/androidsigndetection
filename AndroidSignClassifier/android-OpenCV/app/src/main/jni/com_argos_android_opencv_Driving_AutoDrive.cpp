#include "com_argos_android_opencv_Driving_AutoDrive.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <android/log.h>
#include "lanefinder.h"
#include "carfinder.h"
#include "speeddetector.h"


/**
 * Implementation of the native functions
 */

using namespace std;
using namespace cv;

/**
 * Perform all operations on the processed matrix and only draw the detections on the original source matrix
 */
Mat processed;
/**
 * Hardcoded values to set the ROI considering dimensions as 640x480
 */
Point pts[6] = {
    Point(0, 480),
    Point(0, 250),
    Point(240, 200),
    Point(400, 200),
    Point(640, 250),
    Point(640, 480)
};

void processImage(Mat);
void setROI();
void drawDebugLines(Mat&);
void extractRedPixels(Mat&);
void extractRedPixelsTest(Mat&);
void extractRedPixelsTestCanny(Mat&);
void detectCircles(Mat&);

JNIEXPORT jstring JNICALL Java_com_argos_android_opencv_Driving_AutoDrive_drive
        (JNIEnv* env, jclass, jlong srcMat)
{
    Mat& original = *(Mat*) srcMat;

    processImage(original);
    setROI();
    drawDebugLines(original);

    LaneFinder laneFinder(processed, original);
    return env->NewStringUTF(laneFinder.find());
}

void processImage(Mat image)
{
    cvtColor(image, processed, CV_RGBA2GRAY);
    GaussianBlur(processed, processed, Size(5,5), 0, 0);
    Canny(processed, processed, 200, 300, 3);
}

void setROI()
{
    Mat mask(processed.size(), CV_8U);
    fillConvexPoly(mask, pts, 6, Scalar(255));
    bitwise_and(mask, processed, processed);
}

void drawDebugLines(Mat& original)
{
    line(original, pts[0], pts[1], Scalar(255, 0, 0), 1, CV_AA);
    line(original, pts[1], pts[2], Scalar(255, 0, 0), 1, CV_AA);
    line(original, pts[2], pts[3], Scalar(255, 0, 0), 1, CV_AA);
    line(original, pts[3], pts[4], Scalar(255, 0, 0), 1, CV_AA);
    line(original, pts[4], pts[5], Scalar(255, 0, 0), 1, CV_AA);
    line(original, pts[5], pts[0], Scalar(255, 0, 0), 1, CV_AA);
}

JNIEXPORT jstring JNICALL Java_com_argos_android_opencv_Driving_AutoDrive_detectVehicle
        (JNIEnv* env, jclass, jstring cascadeFilePath, jlong srcMat)
{
    Mat& original = *(Mat*) srcMat;
    const char* javaString = env->GetStringUTFChars(cascadeFilePath, NULL);
    string cascadeFile(javaString);

    CarFinder carFinder(original, original, cascadeFile);
    return env->NewStringUTF(carFinder.find());
}

JNIEXPORT jstring JNICALL Java_com_argos_android_opencv_Driving_AutoDrive_detectSpeedLimit
        (JNIEnv* env, jclass, jlong srcMat)
  {
    Mat& original = *(Mat*) srcMat;
    SpeedDetector speedDetector(original, original);
    extractRedPixels(original);
    detectCircles(original);
    return env->NewStringUTF(speedDetector.detect());
  }

void extractRedPixels(Mat& original)
{
    //medianBlur(original, processed, 3);

    cvtColor(original, processed, COLOR_RGB2HSV);
    Mat lower_red_hue_range;
    Mat upper_red_hue_range;
    inRange(processed, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);
    inRange(processed, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue_range);
    addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, processed);
    GaussianBlur(processed, processed, Size(9, 9), 2, 2);


    //cvtColor(original, original, COLOR_RGBA2GRAY);
}
void extractRedPixelsTest(Mat& original)
{
    //medianBlur(original, original, 3);

    cvtColor(original, original, COLOR_RGB2HSV);
    Mat lower_red_hue_range;
    Mat upper_red_hue_range;
    inRange(original, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue_range);
    inRange(original, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue_range);
    addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, original);
    GaussianBlur(original, original, Size(9, 9), 2, 2);


    //cvtColor(original, original, COLOR_RGBA2GRAY);
}
void extractRedPixelsTestCanny(Mat& original)
{

    cvtColor(original, processed, COLOR_RGBA2GRAY);
    blur(processed, processed, Size(3,3) );
    Canny(processed, processed, 100, 200, 3);
}

void detectCircles(Mat& original)
{
    vector<Vec3f> circles;
    HoughCircles(processed, circles, CV_HOUGH_GRADIENT, 1, processed.rows/4, 100, 20, 15, 0);

    if(circles.size() == 0) return;
     	for(size_t current_circle = 0; current_circle < circles.size(); current_circle++) {
     		Point center(round(circles[current_circle][0]), round(circles[current_circle][1]));
     		int radius = round(circles[current_circle][2]);

     		rectangle( original,
                           cvPoint(center.x - radius, center.y - radius),
                           cvPoint(center.x + radius, center.y + radius),
                           CV_RGB(0, 255, 0), 2, 8
                         );

            putText(original,  "Sign Candidate", cvPoint(center.x + radius + 5, center.y),
                    0, 0.3, CV_RGB(0, 255, 0));
     	}
}

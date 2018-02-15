//
// Created by bauer on 16.11.2017.
//

#ifndef ANDROID_OPENCV_SPEEDDETECTOR_H
#define ANDROID_OPENCV_SPEEDDETECTOR_H


#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace std;
using namespace cv;

class SpeedDetector{

private:

    Mat image;
    Mat output;


public:
     SpeedDetector(Mat processed, Mat original)
     {
         image = processed;
         output = original;
     }

     char* detect(){

        return "S";
     }
};

#endif //ANDROID_OPENCV_SPEEDDETECTOR_H

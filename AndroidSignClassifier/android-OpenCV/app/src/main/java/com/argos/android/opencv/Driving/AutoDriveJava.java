package com.argos.android.opencv.Driving;


import android.content.res.AssetManager;
import android.widget.Toast;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

/**
 * Created by Tobias Bauer on 26.01.18.
 */

public  class AutoDriveJava {

    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "Add_2";
    private static final String [] OUTPUT_NAMES = {"Add_2"};


    public static TensorFlowInferenceInterface initTensorflowModel(AssetManager assetManager){
        TensorFlowInferenceInterface tensorflow = new TensorFlowInferenceInterface(assetManager, "file:///android_asset/frozen_model.pb");
        return tensorflow;
    }

    public static String detectSpeedLimit(Mat srcMat, TensorFlowInferenceInterface tensorflow) {
        Imgproc.medianBlur(srcMat, srcMat, 3);

        Mat hsv_image = new Mat();
        Imgproc.cvtColor(srcMat, hsv_image, Imgproc.COLOR_RGB2HSV);
        //Threshold image keep only red pixels

        Mat lower_red_hue_range = new Mat();
        Mat upper_red_hue_range = new Mat();

        Core.inRange(hsv_image, new Scalar(0, 100, 100), new Scalar(10, 255, 255), lower_red_hue_range);
        Core.inRange(hsv_image, new Scalar(160, 100, 100), new Scalar(179, 255, 255), upper_red_hue_range);



        //Combine Images
        Mat red_hue_image = new Mat();
        Core.addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);

        Imgproc.GaussianBlur(red_hue_image, red_hue_image, new Size(9, 9), 2, 2);
        Mat circles = new Mat();
        Point[] circleList = new Point[0];
        Imgproc.HoughCircles(red_hue_image, circles, Imgproc.CV_HOUGH_GRADIENT, 1, red_hue_image.rows() / 4, 100, 20, 0, 0);
        Mat [] imageROIs = new Mat[10];
        for (int x = 0; x < circles.cols(); x++) {
            double[] c = circles.get(0, x);
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            // circle outline
            int radius = (int) Math.round(c[2]);
            //Imgproc.circle(srcMat, center, radius, new Scalar(0,255,0), 3, 8, 0 );
            Point xCoord = new Point(center.x - radius, center.y - radius);
            Point yCoord = new Point(center.x + radius, center.y + radius);
            Imgproc.rectangle(srcMat, xCoord, yCoord, new Scalar(0, 255, 0), 2);
            //imageROIs[x] = srcMat.submat((int)(center.y - radius), (int)(center.y + radius), (int)(center.x - radius), (int) (center.x + radius));
        }
        ArrayList<Mat> ROIs;
        float [][] result;
        //result = classify(ROIs, tensorflow);
        String out = "[";
        for (int i= 0; i<10; i++){
            //out += result[0][i] + ", ";
        }

        Imgproc.putText(srcMat, out, new Point(10,100), 0, 0.3, new Scalar(0,255,0));

        return "HOLLA";

    }

    public static String detectSpeedLimitLoadedImage(Mat srcMat, TensorFlowInferenceInterface tensorflow) {
        Imgproc.medianBlur(srcMat, srcMat, 3);

        Mat hsv_image = new Mat();
        Imgproc.cvtColor(srcMat, hsv_image, Imgproc.COLOR_BGR2HSV);
        //Threshold image keep only red pixels

        Mat lower_red_hue_range = new Mat();
        Mat upper_red_hue_range = new Mat();

        Core.inRange(hsv_image, new Scalar(0, 100, 100), new Scalar(10, 255, 255), lower_red_hue_range);
        Core.inRange(hsv_image, new Scalar(160, 100, 100), new Scalar(179, 255, 255), upper_red_hue_range);

        //Combine Images
        Mat red_hue_image = new Mat();
        Core.addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);

        Imgproc.GaussianBlur(red_hue_image, red_hue_image, new Size(9, 9), 2, 2);
        Mat circles = new Mat();
        Point[] circleList = new Point[0];
        Imgproc.HoughCircles(red_hue_image, circles, Imgproc.CV_HOUGH_GRADIENT, 1, red_hue_image.rows() / 4, 100, 20, 0, 0);
        ArrayList<Mat> imageROIs = new ArrayList<Mat>();
        for (int x = 0; x < circles.cols(); x++) {
            double[] c = circles.get(0, x);
            Point center = new Point(Math.round(c[0]), Math.round(c[1]));
            // circle outline
            int radius = (int) Math.round(c[2]);
            //Imgproc.circle(srcMat, center, radius, new Scalar(0,255,0), 3, 8, 0 );
            Point xCoord = new Point(center.x - radius, center.y - radius);
            Point yCoord = new Point(center.x + radius, center.y + radius);
            Imgproc.rectangle(srcMat, xCoord, yCoord, new Scalar(0, 255, 0), 2);
            Rect rect = new Rect(xCoord, yCoord);
            imageROIs.add(srcMat.submat(rect));

            srcMat = imageROIs.get(0);
        }
        float [][] result;
        result = classify(imageROIs, tensorflow);
        int y = 820;
        String out = "";
        for (int i= 0; i<10; i++){
            out = Integer.toString((i+1) *10) + "   " + result[0][i];
            Imgproc.putText(srcMat, out, new Point(10,y), 0, 0.7, new Scalar(0,255,0));
            y+=20;
        }
        return "HOLLA";

    }


    private static float[][] classify(ArrayList<Mat> images, TensorFlowInferenceInterface tensorflow) {
        Mat currentimage;
        float [] imagetensor = new float[400];
        ArrayList <float[]> tensors;
        for (int i = 0; i < images.size(); i++){
            currentimage = images.get(i);
            imagetensor = get2020binArray(currentimage);
        }
        /*
        float [] image = new float[400];
            double[] imdoub = {-0.5, -0.5, -0.5, -0.5, -0.5, -0.496078, -0.460784, -0.472549, -0.492157, -0.492157, -0.480392, -0.5, -0.5, -0.5, -0.5, -0.5, -0.476471, -0.492157, -0.492157, -0.496078, -0.5, -0.5, 0.44902, 0.472549, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.484314, -0.5, -0.492157, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.484314, 0.476471, 0.5, 0.5, 0.496078, 0.5, 0.5, 0.5, 0.5, 0.496078, 0.5, 0.5, 0.484314, 0.5, -0.488235, -0.480392, -0.5, -0.5, -0.5, -0.5, 0.464706, 0.488235, 0.5, 0.5, 0.5, 0.480392, 0.5, 0.5, 0.5, 0.5, 0.484314, 0.5, 0.480392, 0.480392, 0.5, -0.5, 0.5, -0.5, -0.5, 0.476471, 0.468627, 0.468627, 0.488235, 0.492157, 0.488235, 0.480392, 0.488235, 0.476471, 0.484314, 0.492157, 0.468627, 0.460784, 0.456863, 0.452941, 0.5, 0.5, 0.5, -0.5, -0.496078, 0.460784, 0.464706, 0.492157, 0.5, -0.496078, -0.5, 0.492157, 0.468627, 0.5, 0.492157, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.480392, -0.484314, -0.488235, 0.5, 0.492157, 0.5, -0.5, -0.445098, -0.476471, 0.5, 0.5, 0.5, 0.5, -0.472549, -0.488235, 0.496078, -0.5, -0.5, -0.456863, 0.5, 0.5, -0.476471, -0.472549, 0.5, 0.5, -0.5, -0.472549, -0.468627, 0.5, 0.5, 0.5, 0.5, -0.452941, -0.464706, 0.5, 0.5, 0.5, -0.496078, -0.468627, 0.5, 0.5, -0.496078, -0.5, 0.480392, 0.476471, -0.5, -0.5, 0.488235, 0.480392, 0.480392, 0.464706, 0.480392, -0.5, -0.5, 0.472549, 0.480392, 0.488235, -0.5, -0.480392, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.468627, 0.5, -0.5, -0.5, 0.5, 0.476471, 0.480392, -0.484314, -0.472549, 0.5, 0.5, -0.5, -0.5, 0.496078, -0.5, -0.492157, -0.5, 0.5, -0.5, -0.492157, -0.488235, 0.5, -0.484314, -0.492157, 0.5, 0.492157, 0.5, -0.496078, -0.472549, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.492157, 0.496078, 0.496078, 0.472549, -0.496078, -0.492157, 0.5, -0.484314, -0.460784, 0.5, 0.496078, 0.5, -0.484314, -0.456863, 0.5, 0.5, -0.5, -0.5, 0.5, -0.488235, -0.5, 0.492157, 0.496078, 0.5, -0.488235, -0.5, 0.5, -0.480392, -0.5, 0.5, 0.464706, 0.496078, -0.5, -0.460784, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.492157, -0.5, -0.492157, -0.5, -0.488235, -0.5, 0.492157, 0.5, -0.5, -0.5, -0.5, -0.5, -0.496078, 0.5, 0.5, 0.5, -0.5, -0.488235, 0.5, 0.492157, 0.480392, -0.496078, -0.484314, -0.5, -0.488235, 0.488235, 0.5, 0.5, 0.492157, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, -0.496078, -0.472549, 0.5, 0.480392, 0.496078, 0.5, 0.5, 0.492157, 0.5, 0.5, 0.5, 0.5, 0.5, 0.492157, 0.472549, 0.488235, 0.5, 0.5, 0.5, 0.5, -0.496078, -0.476471, -0.492157, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.476471, -0.484314, -0.488235, -0.480392, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.480392, 0.5, -0.488235, -0.468627, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.468627, 0.5, 0.5, -0.472549, -0.5, -0.496078, -0.5, -0.5, -0.488235, -0.488235, -0.460784, -0.468627, -0.480392, -0.480392, -0.480392, -0.480392, -0.480392, -0.480392, -0.480392, -0.480392, -0.488235, -0.484314, -0.484314, -0.5};

        */
        float[] result = new float[10];

        float[][] a = new float[10][10];

        // Copy the input data into TensorFlow.
        tensorflow.feed(INPUT_NAME, imagetensor, 1,400);

        // Run the inference call.
        tensorflow.run(OUTPUT_NAMES);

        // Copy the output Tensor back into the output array.
        tensorflow.fetch(OUTPUT_NAME, result);
        a[0] = result;
        return a;
    }

    private static float[] get2020binArray(Mat image){
        Mat img2020 = new Mat();
        Imgproc.resize(image, img2020, new Size(20, 20));
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img2020, imgGray, Imgproc.COLOR_BGR2GRAY);
        Mat imgBinary = new Mat();
        Imgproc.threshold(imgGray, imgBinary, 150, 255, Imgproc.THRESH_BINARY);
        float [] flbuff = new float[400];
        int [] intbuff = new int[400];
        String s = imgBinary.dump();
        double doubhelp;
        int c = 0;
        for (int i= 0; i<20; i++) {
            for (int j = 0; j < 20; j++) {
                doubhelp = imgBinary.get(i, j)[0];
                flbuff[c] = (float) ((doubhelp / 255) - 0.5);
                c++;
            }
        }
        return flbuff;

    }

}

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat calib(Mat &frame)
{
        //Mat frame;
        Mat frameCalibration;

        Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = 480.0497;
        cameraMatrix.at<double>(0, 1) = 0;
        cameraMatrix.at<double>(0, 2) = 349.8985;
        cameraMatrix.at<double>(1, 1) = 482.3921;
        cameraMatrix.at<double>(1, 2) = 229.4727;

        Mat distCoeffs = Mat::zeros(5, 1, CV_64F);
        distCoeffs.at<double>(0, 0) = -0.3218;
        distCoeffs.at<double>(1, 0) = 0.1219;
        distCoeffs.at<double>(2, 0) = 0;
        distCoeffs.at<double>(3, 0) = 0;
        distCoeffs.at<double>(4, 0) = 0;

        Mat view, rview, map1, map2;
        Size imageSize;
        imageSize = frame.size();
        initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(),
                                getOptimalNewCameraMatrix(cameraMatrix, distCoeffs,imageSize, 0, imageSize, 0),
                                imageSize, CV_32FC1, map1, map2);

        remap(frame, frameCalibration, map1, map2, INTER_LINEAR);
        //imshow("Origianl", frame);
        imshow("Calibration", frameCalibration);

    return  frameCalibration;
}
//
// Created by mac on 2019-08-09.
//

#ifndef OPENCV_TEST_WEDDING_H
#define OPENCV_TEST_WEDDING_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
using namespace cv;
using namespace std;


int frameProcess(Mat &init_frame, Mat &frame)
{
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    init_frame = Mat::zeros(frame.size(), CV_8UC1);
    blur(frame, frame, Size(3,3));
    cvtColor(frame, frame, COLOR_BGR2GRAY);
    Canny(frame, frame, 20, 80, 3, false);
    findContours(init_frame, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    RNG rng(0);
    for(int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255));
        drawContours(init_frame, contours, i, color, 2, 8, hierarchy, 0, Point(0, 0));
    }
    return 1;
}

int cannyProcess(Mat &frame, Mat &edges)
{
    cvtColor(frame, edges, COLOR_BGR2GRAY);
    blur(edges, edges, Size(7, 7));
    Canny(edges, edges, 0, 30, 3);
    return 0;
}

#define T 0 //根据实际情况设定固定阈值
Point grayCenter(Mat& img)
{
    Mat img_gray;
    img_gray = img;
    //cvtColor(img, img_gray, COLOR_BGR2GRAY, 0);
    Point Center; //中心点
    int i, j;
    double sumval = 0;
    MatIterator_<uchar> it, end;
    //获取图像各点灰度值总和
    for (it = img_gray.begin<uchar>(), end = img_gray.end<uchar>(); it != end; it++)
    {
        ((*it) > T) ? sumval += (*it) : 0; //小于阈值，取0
    }
    Center.x = Center.y = 0;
    double x = 0, y = 0;
    for (int i = 0; i < img_gray.cols; i++)
    {
        for (int j = 0; j < img_gray.rows; j++)
        {
            double s = img_gray.at<uchar>(j, i); //取当前点灰度值
            if (s < T)
                s = 0;
            x += i * s / sumval;
            y += j * s / sumval;
        }
    }
    Center.x = cvRound(x);
    Center.y = cvRound(y);
    //cout << "rows=" << img_gray.rows << "  cols=" << img_gray.cols << endl;
    cout << "x=" << x << "  y=" << y << endl;
    return Center;
}

int videoShow()
{
    VideoCapture capture(1);
    Mat frame;
    Mat init_frame;
    Mat edges;
    while(1)
    {
       if(!capture.isOpened())
       {
            return -1;
       }
       capture >> frame;
       //cannyProcess(frame, edges);
       //frameProcess(init_frame, frame);
       //grayCenter(frame);
       imshow("show", frame);
       waitKey(30);
    }
    return 0;
}

void drawCross(Mat img, Point point,Scalar color,int size, int thickness, int lineType=8)
{
    //drawLine(img, Point(point.x - size/2, point.y), Point(point.x+size/2, point.y), color, thickness, 8, 0);
    //drawLine(img, Point(point.x, point.y-size/2), Point(point.x, point.y+size/2), color, thickness, 8, 0);
    //绘制横线
    line(img, Point(point.x - size/2, point.y), Point(point.x+size/2, point.y), color, thickness, 8, 0);
    //绘制竖线
    line(img, Point(point.x, point.y-size/2), Point(point.x, point.y+size/2),color, thickness, 8, 0);
}

bool numSort(vector<Point> a, vector<Point> b)
{
    return a.size() < b.size();

}

void StegerLine()
{
    Mat img0 = imread("/Users/mac/Desktop/test1.jpg", 1);
    Mat img;
    cvtColor(img0, img0, COLOR_BGR2GRAY);
    img = img0.clone();
    Mat mask =Mat::zeros(img.size(), CV_8UC1);

    //高斯滤波
    img.convertTo(img, CV_32FC1);
    GaussianBlur(img, img, Size(0, 0), 6, 6);

    //一阶偏导数
    Mat m1, m2;
    m1 = (Mat_<float>(1, 2) << 1, -1);  //x偏导
    m2 = (Mat_<float>(2, 1) << 1, -1);  //y偏导

    Mat dx, dy;
    filter2D(img, dx, CV_32FC1, m1);
    filter2D(img, dy, CV_32FC1, m2);

    //二阶偏导数
    Mat m3, m4, m5;
    m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //二阶x偏导
    m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //二阶y偏导
    m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //二阶xy偏导

    Mat dxx, dyy, dxy;
    filter2D(img, dxx, CV_32FC1, m3);
    filter2D(img, dyy, CV_32FC1, m4);
    filter2D(img, dxy, CV_32FC1, m5);

    //hessian矩阵
    double maxD = -1;
    int imgcol = img.cols;
    int imgrow = img.rows;
    vector<double> Pt;
    for (int i=0;i<imgcol;i++)
    {
        for (int j=0;j<imgrow;j++)
        {
            if (img0.at<uchar>(j,i)>200)
            {
                Mat hessian(2, 2, CV_32FC1);
                hessian.at<float>(0, 0) = dxx.at<float>(j, i);
                hessian.at<float>(0, 1) = dxy.at<float>(j, i);
                hessian.at<float>(1, 0) = dxy.at<float>(j, i);
                hessian.at<float>(1, 1) = dyy.at<float>(j, i);

                Mat eValue;
                Mat eVectors;
                eigen(hessian, eValue, eVectors);

                double nx, ny;
                double fmaxD = 0;
                if (fabs(eValue.at<float>(0,0))>= fabs(eValue.at<float>(1,0)))  //求特征值最大时对应的特征向量
                {
                    nx = eVectors.at<float>(0, 0);
                    ny = eVectors.at<float>(0, 1);
                    fmaxD = eValue.at<float>(0, 0);
                }
                else
                {
                    nx = eVectors.at<float>(1, 0);
                    ny = eVectors.at<float>(1, 1);
                    fmaxD = eValue.at<float>(1, 0);
                }

                double t = -(nx*dx.at<float>(j, i) + ny*dy.at<float>(j, i)) / (nx*nx*dxx.at<float>(j,i)+2*nx*ny*dxy.at<float>(j,i)+ny*ny*dyy.at<float>(j,i));

                if (fabs(t*nx)<=0.5 && fabs(t*ny)<=0.5)
                {
                    Pt.push_back(i);
                    Pt.push_back(j);
                }
            }
        }
    }

    vector<Point> max;
    for (int k = 0; k < Pt.size()/2; k++)
    {
        Point rpt;
        rpt.x = Pt[2 * k + 0];
        rpt.y = Pt[2 * k + 1];
        //circle(img0, rpt, 1, Scalar(255, 0, 0));
        circle(mask, rpt, 1, Scalar(255, 0, 0));
        //cout << rpt <<endl;
        max.push_back(rpt);
    }

    int max1=max[0].y;//假设最大值是第一个数
    int temp=0;
    for(int i = 1; i < max.size(); i++)
    {
        if(max[i].y > max1)
        {
            temp = i;//保存最大值的下标
            max1 = max[i].y;
        }
        //cout<<"最大值是："<< max <<" 最大值坐标为："<< max[temp] <<endl;//输出最大值及下标
    }
    circle(mask, max[temp], 6, Scalar(255, 255, 255));
    cout << max[temp] << endl;

    //drawCross(mask, grayCenter(mask), Scalar(100, 0, 0), 3, 2, 8);
    imshow("result", mask);
    waitKey(0);
}

int videoRead()
{
    VideoCapture cap("/Users/mac/Downloads/open.mp4");

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while(1){

        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
            break;
        //grayCenter(frame);
        drawCross(frame, grayCenter(frame), Scalar(255, 0, 0), 3, 2, 8);
        // Display the resulting frame
        imshow( "Frame", frame );
        // Press  ESC on keyboard to exit
        char c=(char)waitKey(25);
        if(c==27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    destroyAllWindows();

    return 1;
}

int jpgRead()
{
    Mat img = imread("/Users/mac/Desktop/test1.jpg", 1);
    if(!img.data)  //or == if(src.empty())
    {
        cout<<"Could not open or find the image"<< endl;
        return -1;
    }
    // 创建窗口
    namedWindow("Display", WINDOW_AUTOSIZE);
    //显示图像
    drawCross(img, grayCenter(img), Scalar(255, 0, 0), 3, 2, 8);
    imshow("Display", img);

    //暂停，等待按键结束
    waitKey(0);
    return 0;
}

bool LessSort (Point &a,Point &b) { return (a.x<b.x); }

int videoSteger()
{
    VideoCapture cap(1);

    cap.set(CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CAP_PROP_FRAME_HEIGHT, 240);

    // Check if camera opened successfully
    if(!cap.isOpened()){
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    while(1){

        Mat frame;
        // Capture frame-by-frame
        cap >> frame;
        if (frame.empty()){
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        Mat img0 = frame;
        Mat img;
        cvtColor(img0, img0, COLOR_BGR2GRAY);
        img = img0.clone();
        //Mat mask =Mat::zeros(img.size(), CV_8UC1);
        Mat mask =Mat::zeros(img.size(), CV_32FC1);
        //高斯滤波
        img.convertTo(img, CV_32FC1);
        GaussianBlur(img, img, Size(0, 0), 6, 6);

        //一阶偏导数
        Mat m1, m2;
        m1 = (Mat_<float>(1, 2) << 1, -1);  //x偏导
        m2 = (Mat_<float>(2, 1) << 1, -1);  //y偏导

        Mat dx, dy;
        filter2D(img, dx, CV_32FC1, m1);
        filter2D(img, dy, CV_32FC1, m2);

        //二阶偏导数
        Mat m3, m4, m5;
        m3 = (Mat_<float>(1, 3) << 1, -2, 1);   //二阶x偏导
        m4 = (Mat_<float>(3, 1) << 1, -2, 1);   //二阶y偏导
        m5 = (Mat_<float>(2, 2) << 1, -1, -1, 1);   //二阶xy偏导

        Mat dxx, dyy, dxy;
        filter2D(img, dxx, CV_32FC1, m3);
        filter2D(img, dyy, CV_32FC1, m4);
        filter2D(img, dxy, CV_32FC1, m5);

        //hessian矩阵
        double maxD = -1;
        int imgcol = img.cols;
        int imgrow = img.rows;
        vector<double> Pt;
        for (int i=0;i<imgcol;i++)
        {
            for (int j=0;j<imgrow;j++)
            {
                if (img0.at<uchar>(j,i)>200)
                {
                    Mat hessian(2, 2, CV_32FC1);
                    hessian.at<float>(0, 0) = dxx.at<float>(j, i);
                    hessian.at<float>(0, 1) = dxy.at<float>(j, i);
                    hessian.at<float>(1, 0) = dxy.at<float>(j, i);
                    hessian.at<float>(1, 1) = dyy.at<float>(j, i);

                    Mat eValue;
                    Mat eVectors;
                    eigen(hessian, eValue, eVectors);

                    double nx, ny;
                    double fmaxD = 0;
                    if (fabs(eValue.at<float>(0,0))>= fabs(eValue.at<float>(1,0)))  //求特征值最大时对应的特征向量
                    {
                        nx = eVectors.at<float>(0, 0);
                        ny = eVectors.at<float>(0, 1);
                        fmaxD = eValue.at<float>(0, 0);
                    }
                    else
                    {
                        nx = eVectors.at<float>(1, 0);
                        ny = eVectors.at<float>(1, 1);
                        fmaxD = eValue.at<float>(1, 0);
                    }

                    double t = -(nx*dx.at<float>(j, i) + ny*dy.at<float>(j, i)) / (nx*nx*dxx.at<float>(j,i)+2*nx*ny*dxy.at<float>(j,i)+ny*ny*dyy.at<float>(j,i));

                    if (fabs(t*nx)<=0.5 && fabs(t*ny)<=0.5)
                    {
                        Pt.push_back(i);
                        Pt.push_back(j);
                    }
                }
            }
        }

        vector<Point> max;
        for (int k = 0; k < Pt.size()/2; k++)
        {
            Point rpt;
            rpt.x = Pt[2 * k + 0];
            rpt.y = Pt[2 * k + 1];
            //circle(img0, rpt, 1, Scalar(255, 0, 0));
            circle(mask, rpt, 1, Scalar(255, 0, 0));
            //cout << rpt <<endl;
            max.push_back(rpt);
        }


        sort(max.begin(), max.end(), LessSort);//升序排列

        auto it = max.begin();
        while(it != max.end())
        {
            if(it -> y < 60 || it -> y > 160)
                it = max.erase(it);
            else
                ++it;
        }

        for (int i = 0;i < max.size();i++)
        {
            cout << max.back() << endl;
            circle(mask, max.back(), 6, Scalar(255, 255, 255));
        }


//        int max1 = max[0].x;//假设最大值是第一个数
//        int temp = 0;
//        for(int i = 1; i < max.size(); i++)
//        {
//            if(max[i].x > max1)
//            {
//                temp = i;//保存最大值的下标
//                max1 = max[i].x;
//            }
//            //cout<<"最大值是："<< max <<" 最大值坐标为："<< max[temp] <<endl;//输出最大值及下标
//        }
//
//        circle(mask, max[temp], 6, Scalar(255, 255, 255));
//        cout << max[temp] << endl;
        //drawCross(mask, grayCenter(mask), Scalar(100, 0, 0), 3, 2, 8);
        imshow("result", mask);
        waitKey(30);
    }
    return 1;
}

#endif //OPENCV_TEST_WEDDING_H

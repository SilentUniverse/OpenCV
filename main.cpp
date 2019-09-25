#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
//#include "wedding.h"
#include "main.h"

using namespace std;
using namespace cv;
//using namespace cv::xfeatures2d;

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

    int count = 0;

    for (;;)
    {

        Mat frame;
        // Capture frame-by-frame
        cap.read(frame);
        if (frame.empty()){
            cout << "ERROR! blank frame grabbed\n";
            break;
        }
        //imshow("org", frame);
        Mat img0;
        img0 = calib(frame);
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
        //检测是否为空
        if(Pt.empty())
            continue;

        //保存找到的所有点
        vector<Point> max;
        for (int k = 0; k < Pt.size()/2; k++)
        {
            Point rpt;
            rpt.x = static_cast<int>(Pt[2 * k + 0]);
            rpt.y = static_cast<int>(Pt[2 * k + 1]);
            //circle(img0, rpt, 1, Scalar(255, 0, 0));
            circle(mask, rpt, 1, Scalar(255, 0, 0));//在纯黑的背景的图像画出找到的点
            //cout << rpt <<endl;
            if(rpt.x > 0||rpt.y > 0)
            {
                max.push_back(rpt);
            }else{
                continue;
            }
        }

        sort(max.begin(), max.end(), LessSort);//升序排列

        //删除不在图像中间区域的点
        auto it = max.begin();
        while(it != max.end())
        {
            if(it -> y < 100 || it -> y > 140)
                it = max.erase(it);
            else
                ++it;
        }

        //圈出 y 值最大的点
        count += 1 ;
//        for (auto i: max)
//        {
//        }
        circle(mask, max.back(), 6, Scalar(255, 255, 255));


        time_t nowtime;
        nowtime = time(NULL); //获取日历时间

        struct tm *local;
        local = localtime(&nowtime);  //获取当前系统时间

        char buf[80];
        strftime(buf, 80, "%Y-%m-%d %H:%M:%S", local);

        if(max.back().x > 320||max.back().y > 240||max.back().x <= 0||max.back().y <= 0)
        {
            continue;
        }else{
            if(count % 2 == 0)
            {
                cout << buf <<"  "<<"x: "<<max.back().x <<" "<<"y: "<< max.back().y << endl;
            }else{
                continue;
            }
        }
        //释放 vector 内存
        vector<Point>temp;
        temp.swap(max);
        vector<double>temp2;
        temp2.swap(Pt);

        imshow("result", mask);
        if (waitKey(5) >= 0)
            break;
    }
    cap.release();
    return 1;
}


int main()
{
    //featured_based();
    //videoRead();
    //jpgRead();
    //videoShow();
    videoSteger();
    //test();
    //timeStamp();
    return 1;
}


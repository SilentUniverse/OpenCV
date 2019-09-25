//
// Created by mac on 2019-08-08.
//

#ifndef OPENCV_TEST_FEATURE_H
#define OPENCV_TEST_FEATURE_H

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int featured_based()
{
    //Create SIFT class pointer
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    Mat img_1 = imread("1.png");
    Mat img_2 = imread("4.png");
    //Detect the keypoints
    vector<KeyPoint> keypoints_1, keypoints_2;
    f2d->detect(img_1, keypoints_1);
    f2d->detect(img_2, keypoints_2);
    //Calculate descriptors (feature vectors)
    Mat descriptors_1, descriptors_2;
    f2d->compute(img_1, keypoints_1, descriptors_1);
    f2d->compute(img_2, keypoints_2, descriptors_2);
    //Matching descriptor vector using BFMatcher
    BFMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);
    //绘制匹配出的关键点
    Mat img_matches;
    //drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
    //imshow("match", img_matches);

    //计算匹配结果中距离最大和距离最小值
    double min_dist = matches[0].distance, max_dist = matches[0].distance;
    for (int m = 0; m < matches.size(); m++) {
        if (matches[m].distance<min_dist) {
            min_dist = matches[m].distance;
        }
        if (matches[m].distance>max_dist) {
            max_dist = matches[m].distance;
        }
    }
    cout << "min dist=" << min_dist << endl;
    cout << "max dist=" << max_dist << endl;
    //筛选出较好的匹配点
    vector<DMatch> goodMatches;
    for (int m = 0; m < matches.size(); m++) {
        // 一般用 0.5，0.6，要少一点就往下调整
        if (matches[m].distance < 0.15*max_dist) {
            goodMatches.push_back(matches[m]);
        }
    }
    cout << "The number of good matches:" <<goodMatches.size()<< endl;
    Mat img_out;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, goodMatches, img_out,Scalar::all(-1));
    imshow("good Matches",img_out);

    //RANSAC匹配过程
    vector<DMatch> m_Matches;
    m_Matches = goodMatches;
    int ptCount = goodMatches.size();
//    if (ptCount < 100) {
//        cout << "Don't find enough match points" << endl;
//        return 0;
//    }

    //坐标转换为float类型
    vector <KeyPoint> RAN_KP1, RAN_KP2;
    //size_t是标准C库中定义的，应为unsigned int，在64位系统中为long unsigned int,在C++中为了适应不同的平台，增加可移植性。
    for (size_t i = 0; i < m_Matches.size(); i++) {
        RAN_KP1.push_back(keypoints_1[goodMatches[i].queryIdx]);
        RAN_KP2.push_back(keypoints_2[goodMatches[i].trainIdx]);
        //RAN_KP1是要存储img01中能与img02匹配的点
        //goodMatches存储了这些匹配点对的img01和img02的索引值
    }
    //坐标变换
    vector <Point2f> p01, p02;
    for (size_t i = 0; i < m_Matches.size(); i++) {
        p01.push_back(RAN_KP1[i].pt);
        p02.push_back(RAN_KP2[i].pt);
    }
    /*vector <Point2f> img1_corners(4);
	img1_corners[0] = Point(0,0);
	img1_corners[1] = Point(img_1.cols,0);
	img1_corners[2] = Point(img_1.cols, img_1.rows);
	img1_corners[3] = Point(0, img_1.rows);
	vector <Point2f> img2_corners(4);*/
    ////求转换矩阵
    //Mat m_homography;
    //vector<uchar> m;
    //m_homography = findHomography(p01, p02, RANSAC);//寻找匹配图像
    //求基础矩阵 Fundamental,3*3的基础矩阵
    vector<uchar> RansacStatus;
    Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);
    //重新定义关键点RR_KP和RR_matches来存储新的关键点和基础矩阵，通过RansacStatus来删除误匹配点
    vector <KeyPoint> RR_KP1, RR_KP2;
    vector <DMatch> RR_matches;
    int index = 0;
    for (size_t i = 0; i < m_Matches.size(); i++) {
        if (RansacStatus[i] != 0) {
            RR_KP1.push_back(RAN_KP1[i]);
            RR_KP2.push_back(RAN_KP2[i]);
            m_Matches[i].queryIdx = index;
            m_Matches[i].trainIdx = index;
            RR_matches.push_back(m_Matches[i]);
            index++;
        }
    }
    cout << "RANSAC后匹配点数" <<RR_matches.size()<< endl;
    Mat img_RR_matches;
    drawMatches(img_1, RR_KP1, img_2, RR_KP2, RR_matches, img_RR_matches);
    imshow("After RANSAC",img_RR_matches);
}



#endif //OPENCV_TEST_FEATURE_H

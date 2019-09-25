#include "opencv2/opencv.hpp"
#include <time.h>

using namespace cv;
using namespace std;

int countFrame()
{
    // Start default camera
    VideoCapture video(1);

    // With webcam get(CV_CAP_PROP_FPS) does not work.
    // Let's see for ourselves.

    double fps = video.get(CAP_PROP_FPS);
    // If you do not care about backward compatibility
    // You can use the following instead for OpenCV 3
    // double fps = video.get(CAP_PROP_FPS);
    cout << "Frames per second using video.get(CV_CAP_PROP_FPS) : " << fps << endl;


    // Number of frames to capture
    int num_frames = 120;

    // Start and end times
    time_t start, end;

    // Variable for storing video frames
    Mat frame;

    cout << "Capturing " << num_frames << " frames" << endl ;

    // Start time
    time(&start);

    // Grab a few frames
    for(int i = 0; i < num_frames; i++)
    {
        video >> frame;
    }

    // End Time
    time(&end);

    // Time elapsed
    double seconds = difftime (end, start);
    cout << "Time taken : " << seconds << " seconds" << endl;

    // Calculate frames per second
    fps  = num_frames / seconds;
    cout << "Estimated frames per second : " << fps << endl;

    // Release video
    video.release();
    return 1;
}


int timeStamp()
{
    while(1)
    {
        VideoCapture cap(1);
//        double timestamps = cap.get(CAP_PROP_POS_MSEC);
//        cout << timestamps << endl;
//        int frame_num = cap.get(CAP_PROP_POS_FRAMES);
//        std::cout<<"Frame Num : "<<frame_num<<std::endl;

        time_t nowtime;
        nowtime = time(NULL); //获取日历时间
        //cout << nowtime << endl;  //输出nowtime

        struct tm *local;
        local=localtime(&nowtime);  //获取当前系统时间

        char buf[80];
        strftime(buf,80,"%Y-%m-%d %H:%M:%S",local);
        cout << buf << endl;

    }
    return 1;
}

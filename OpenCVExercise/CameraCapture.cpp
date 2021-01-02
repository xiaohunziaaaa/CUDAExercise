/*#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(){
    VideoCapture cap(0);
    if(cap.isOpened()==false){
        cout<<"Can not find camera!"<<endl;
        return -1;
    }

    double fps = cap.get(CV_CAP_PROP_FPS);
    cout << "Frames per seconds : " << fps << endl;
    cout<<"Press Q to Quit" <<endl;
    String winName = "Webcam Video";
    namedWindow(winName);

    while(true){
        Mat frame;
        bool flag = cap.read(frame);

        imshow(winName, frame);
        if(waitKey(1)=='q'){
            break;
        }
    }

    return 0;

}*/
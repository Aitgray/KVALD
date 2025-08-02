#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat img(100, 100, CV_8UC3, cv::Scalar(0, 0, 255));
    std::cout << "OpenCV test successful!" << std::endl;
    return 0;
}

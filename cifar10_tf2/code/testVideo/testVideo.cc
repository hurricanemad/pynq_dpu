#include <opencv2/opencv.hpp>


int main(int argc, char* argv[]){
	cv::String strImagePath = "..//..//test_images//cat_438.png";
	
	cv::Mat matSourceImage = cv::imread(strImagePath, -1);
	
	cv::imshow("SourceImage", matSourceImage);
	cv::waitKey(-1);
}

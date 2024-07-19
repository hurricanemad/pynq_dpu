// SPDX-License-Identifier: MIT
//
// based on Vitis AI 1.4 demo code
// Author: Hui Shen(Doxxxx)
// Date: June, 2024

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include "common.h"

/*header file OpenCV for image processing*/
#include <opencv2/opencv.hpp>
using namespace std;



GraphInfo shapes;

/**
 * @brief Cut off the float number to unsigned char
 *
 * @param fInputV input value
 *
 * @return result value
 */

uchar CutoffValue(float fInputV){
   if(fInputV > 255.0f){
      return 255;
   }else if(fInputV < 0.0f){
      return 0;
   }else{
      return static_cast<uchar>(fInputV + 0.5f);
   }
}

/**
 * @brief Convert processed image to result image
 *
 * @param matSrcImage the image output by the DPU net
 * @param matDstImage the result image
 *
 * @return None
 */

void convertImage(cv::Mat& matSrcImage, cv::Mat& matProcessedImage, float fScale){
     float* pscSrcImage = matSrcImage.ptr<float>(0);
     uchar* pucProcessedImage = matProcessedImage.ptr<uchar>(0);

     int nSize = matSrcImage.rows * matSrcImage.cols;

     for(int n=0; n< nSize; n++){
        pucProcessedImage[n] = CutoffValue((pscSrcImage[n])*255.0f);
     }

}


int main(int argc, char* argv[])
{
   // Check args
   if (argc != 3) {
     cout << "Usage: run_cnn model_pathName camera_id" << endl;
     return -1;
    }


    auto graph = xir::Graph::deserialize(argv[1]);
    auto subgraph = get_dpu_subgraph(graph.get());
    CHECK_EQ(subgraph.size(), 1u)
      << "CNN should have one and only one dpu subgraph.";
    LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();
    

    /*create runner*/
    auto runner = vart::Runner::create_runner(subgraph[0], "run");
    // ai::XdpuRunner* runner = new ai::XdpuRunner("./");


    /* get in/out tensors shape*/
    auto outputTensors = runner->get_output_tensors();
    auto inputTensors = runner->get_input_tensors();
    int inputCnt = inputTensors.size();
    int outputCnt = outputTensors.size();
    TensorShape inshapes[inputCnt];
    TensorShape outshapes[outputCnt];
    shapes.inTensorList = inshapes;
    shapes.outTensorList = outshapes;
    getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);


    /* get in/out tensors and dims*/
    auto out_dims = outputTensors[0]->get_shape();
    auto in_dims = inputTensors[0]->get_shape();

    auto input_scale = get_input_scale(inputTensors[0]);
    auto output_scale = get_output_scale(outputTensors[0]);

    /*get shape info*/
    int outSize = shapes.outTensorList[0].size;
    int inSize = shapes.inTensorList[0].size;
    int inHeight = shapes.inTensorList[0].height;
    int inWidth = shapes.inTensorList[0].width;
    int outHeight = shapes.outTensorList[0].height;
    int outWidth = shapes.outTensorList[0].width;
    int batchSize = in_dims[0];
    
    //for debug
    //cout << "OUT  dims " << out_dims  << endl;
    cout << "OUT  size " << outSize   << endl;
    //cout << "IN   dims " << in_dims   << endl;
    cout << "IN   size " << inSize    << endl;
    cout << "IN Height " << inHeight  << endl;
    cout << "IN Width  " << inWidth   << endl;
    cout << "batchSize " << batchSize << endl;
    cout << "input scale is:" << input_scale << endl;
    cout << "output scale is:" << output_scale << endl;
   
    std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
    
    //the SRNet model input variable
    float* imageInputs = new float[inSize * batchSize];
    
    //the SRNet model output variable
    float* SRResult = new float[batchSize * outSize];
    std::vector<vart::TensorBuffer*> inputsPtr, outputsPtr;
    std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

    double dAveragePSNR = 0.0;
    
    //Check the model batch size, if the input batch size is not equal 1, send an alarm to user
    if(batchSize != 1){
    	cerr << "The output dimensions are not compatible with the program!" <<endl;
	return -2;
    }

    in_dims[0] = 1;
    out_dims[0] = batchSize;

    int nCameraID = atof(argv[2]);
    //OpenCV video capture class
    cv::VideoCapture vpCamera(nCameraID);

    //Set the realtime video stream size
    vpCamera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    vpCamera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    
    //send an alarm to user, if the video capture don't open.
    if(!vpCamera.isOpened()){
    	std::cerr << "Error! Camera "<< nCameraID << " is unable to open!" <<endl;
    	return -1;
    }
    
    //The classes using to process
    cv::Mat matSrcImage, matYCrCbImage;
    vector<cv::Mat>vmatYCrCbChannels;
    cv::Mat matInputImage;
    cv::Mat matProcessedImage;
    cv::Mat matResultYImage;
    cv::Mat matResultYCrCbImage;
    cv::Mat matResultImage;
    
    //Initialize a window
    cv::namedWindow("Camera", 1);
    
    //The time variables
    double dStartPreprocess, dPreprocessT;
    double dStartCNNPreprocess, dCNNT;
    double dStartPostProcess, dPostProcessT; 

    while(true){
	//Start the preprocessing timing
	dStartPreprocess = static_cast<double>(cv::getTickCount());
    	//Capture the source image
	vpCamera >> matSrcImage;
        cout << "LINE "<< __LINE__ << endl;
	//Convert the source image to YCrCb Channel image
        cv::cvtColor(matSrcImage, matYCrCbImage, cv::COLOR_BGR2YCrCb);
	cv::split(matYCrCbImage, vmatYCrCbChannels);
        cout << "LINE "<< __LINE__ << endl;
        //Save the Y channel image
        cv::imwrite("TestImage.bmp", vmatYCrCbChannels[0]);
	
	//Upscale the Y channel image 
	for(int n=0; n < 3; n++){
	   cv::resize(vmatYCrCbChannels[n], vmatYCrCbChannels[n], cv::Size(matSrcImage.cols*2, matSrcImage.rows*2), 0, 0, cv::INTER_CUBIC);
	}

        //cout << "LINE "<< __LINE__ << endl;
	cv::Mat matY = vmatYCrCbChannels[0];
	//define the input image
	matInputImage = cv::Mat::zeros(inHeight, inWidth, CV_8UC1);
        //cout << "LINE "<< __LINE__ << endl;
        
	//define the parameters, which can convert the input image to SRNet network
        int nRx = inWidth > matY.cols?(inWidth - matY.cols)/2: 0;
        int nRy = inHeight > matY.rows?(inHeight - matY.rows)/2: 0;
	int nRWidth = inWidth > matY.cols? matY.cols:inWidth;
	int nRHeight = inHeight > matY.rows? matY.rows:inHeight;
        
	//for debug
	//cout << "LINE "<< __LINE__ << endl;
	//cout << "nRx:" << nRx <<endl;
	//cout << "nRy:" << nRy <<endl;
	//cout << "nRWidth:" << nRWidth <<endl;
	//cout << "nRHeight:" << nRHeight <<endl;

        cv::imwrite("TestYImage.bmp", matY);

        //Copy the source image to model input image
        matY.copyTo(matInputImage(cv::Rect(nRx, nRy, nRWidth, nRHeight)));
        
	//preprocess the model input image
        int r, c;
        for(r = 0; r < inHeight; r++){
	   uchar* pucInputImage = matInputImage.ptr<uchar>(r);
	   for(c = 0; c < inWidth; c++){
	      imageInputs[r*inWidth + c] = pucInputImage[c]/255.0f;
	   }
	}
        //cout << "LINE "<< __LINE__ << endl;
        
	//stop the preprocess timing
        dPreprocessT = (static_cast<double>(cv::getTickCount()) - dStartPreprocess)/ cv::getTickFrequency();
        
	//output the time
        cout << "The preprocess time is:" << dPreprocessT <<endl;

        //Start the model processing timing
        dStartCNNPreprocess = static_cast<double>(cv::getTickCount());

        batchTensors.clear();
        inputs.clear();
	outputs.clear();

	//cout << "LINE "<< __LINE__ << endl;

        /* in/out tensor refactory for batch inout/output */
        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
            xir::Tensor::create(inputTensors[0]->get_name(), in_dims,
                            xir::DataType{xir::DataType::FLOAT, 32})));

        //cout << "LINE "<< __LINE__ << endl;

	//input the preprocess image to model input
        inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
            imageInputs, batchTensors.back().get()));

	//cout << "LINE "<< __LINE__ << endl;

        batchTensors.push_back(std::shared_ptr<xir::Tensor>(
           xir::Tensor::create(outputTensors[0]->get_name(), out_dims,
                            xir::DataType{xir::DataType::FLOAT, 32})));

        //cout << "LINE "<< __LINE__ << endl;

	//load the model output to result image variable
        outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(
           SRResult, batchTensors.back().get()));

	//cout << "LINE "<< __LINE__ << endl;

	/*tensor buffer input/output */
        inputsPtr.clear();
        outputsPtr.clear();
        inputsPtr.push_back(inputs[0].get());
        outputsPtr.push_back(outputs[0].get());
        
	//cout << "LINE "<< __LINE__ << endl;

        /*run*/
	//run the SRNet network
        auto job_id = runner->execute_async(inputsPtr, outputsPtr);
        runner->wait(job_id.first, -1);
        
	//cout << "LINE "<< __LINE__ << endl;
        
	//Stop the model processing timing
        dCNNT = (static_cast<double>(cv::getTickCount()) - dStartCNNPreprocess) / cv::getTickFrequency();
        //Output the model processing timing  
        cout << "The CNN process time is:" << dCNNT <<endl;
        //Start the postprocessing timing
        dStartPostProcess = static_cast<double>(cv::getTickCount());

        //Convert the Result variable to model output image
        matProcessedImage = cv::Mat(outHeight, outWidth, CV_32FC1, &(SRResult[0]));
        matResultYImage = cv::Mat(outHeight, outWidth, CV_8UC1);
        
	//cout << "LINE "<< __LINE__ << endl;
        
	//Convert the model output image to result image
        convertImage(matProcessedImage, matResultYImage, output_scale);

	//Set the parameters that convert result image from the model output size to display size
        nRx = outWidth > matY.cols?(outWidth - matY.cols)/2: 0;
        nRy = outHeight > matY.rows?(outHeight - matY.rows)/2: 0;
        nRWidth = outWidth > matY.cols? matY.cols:outWidth;
        nRHeight = outHeight > matY.rows? matY.rows:outHeight;

        //cout << "LINE "<< __LINE__ << endl;

        //Convert the result image size
        vmatYCrCbChannels[0] = matResultYImage(cv::Rect(nRx, nRy, nRWidth, nRHeight));
	//Merge the YCrCb channel image
        cv::merge(vmatYCrCbChannels, matResultYCrCbImage);
	//Convert the YCrCb image to Resultimage
        cv::cvtColor(matResultYCrCbImage, matResultImage, cv::COLOR_YCrCb2BGR);
        //cout << "LINE "<< __LINE__ << endl;
        
	//Stop the postprocessing timing
        dPostProcessT = (static_cast<double>(cv::getTickCount())- dStartPostProcess)/ cv::getTickFrequency();
        //Output the postprocessing timing
        cout << "The postprocess time is:" << dPostProcessT <<endl;


	if(matSrcImage.empty() == false){
           //display the result image
	   cv::imshow("Camera", matResultImage);	
	}

	int nKey = cv::waitKey(30);

	if(nKey == int('q')){
	   break;
	}
    }
    vpCamera.release();
    cv::destroyAllWindows();

    delete[] imageInputs;
    delete[] SRResult;

    return 0;
}


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-License-Identifier: MIT

# Author: Dox
# date:   16 Jun 2024

print(" ")
print("SRNet CNN (pre-trained with Medical Image) fine-tuned to kvasir Dataset, in Pytorch")
print(" ")

# ***********************************************************************
# Import Packages
# ***********************************************************************
import os
import time
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import sys


# ***********************************************************************
# input file names
# ***********************************************************************
srcnet_xmodel  = os.path.join("./SRCNN_pt", "SRCNN_pt.xmodel")


# ***********************************************************************
# Prepare the Overlay and load the "SRNet.xmodel"
# ***********************************************************************
from pynq_dpu import DpuOverlay
overlay = DpuOverlay("dpu.bit")
overlay.load_model(srcnet_xmodel)

dpu = overlay.runner
inputTensors = dpu.get_input_tensors()
outputTensors = dpu.get_output_tensors()
shapeIn = tuple(inputTensors[0].dims)
shapeOut = tuple(outputTensors[0].dims)
outputSize = int(outputTensors[0].get_data_size() / shapeIn[0])
print("shapeIn   : {}".format(shapeIn))
print("shapeOut  : {}".format(shapeOut))
print("outputSize: {}".format(outputSize))

# allocate some buffers that will be re-used multiple times
output_data = [np.empty(shapeOut, dtype=np.float32, order="C")]
input_data = [np.empty(shapeIn, dtype=np.float32, order="C")]
image = input_data[0]

#normalize the input images
def Normalize(image):
    x_test  = np.asarray(image)
    x_test = x_test.astype(np.float32)
    x_test = x_test/255.0
    return x_test

#preprocess the input images
def preprocess_fn(srcimage):
    print("Image width is:{:2d}, Image height is:{:2d}".format(srcimage.shape[1], srcimage.shape[0]))
    rimage = cv.resize(srcimage, (srcimage.shape[1]*2, srcimage.shape[0]*2), cv.INTER_CUBIC)
    imageYCrCb = cv.cvtColor(rimage, cv.COLOR_BGR2YCrCb)

    imageY = np.asarray(imageYCrCb[:, :, 0])
    processimage = np.zeros((shapeIn[1], shapeIn[2], 1), np.uint8)
    
    #define the parameters that can convert the input images into network
    ly = round((shapeIn[1] - imageY.shape[0])/2) if shapeIn[1] > imageY.shape[0] else 0
    rx = round((shapeIn[2] - imageY.shape[1])/2) if shapeIn[2] > imageY.shape[1] else 0
    ih = imageY.shape[0] if shapeIn[1] > imageY.shape[0] else shapeIn[1]
    iw = imageY.shape[1] if shapeIn[2] > imageY.shape[1] else shapeIn[2] 
    
    #for debug
    print(shapeIn)
    print("Image shape:{:2d}, {:2d}".format(imageY.shape[0], imageY.shape[1]))
    print("rt:{:2d}, ih:{:2d}, lt:{:2d}, iw:{:2d}".format(rx, iw, ly, ih))
    
    #convert the source image to the network input size image
    processimage[ly: ly+ih, rx: rx+iw, 0] = imageY

    #Save the Y channel image of input image
    #FileName = image_filename.split('.', -1)[-2]
    #SaveFileName = FileName.split('/',-1)[4]+'Src.bmp'
    #image2 = imageY.astype(np.float32)
    #cv2.imwrite(SaveFileName, imageY)
    
    #Normailize the input images
    image2 = Normalize(processimage) #added by me for ResNet18

    return image2, imageYCrCb

def postprocess_fn(outputimage, outimagew, outimageh):
    
    #define the parameters that can convert the network output image to video size
    lt = round((shapeOut[2] - outimagew)/2) if shapeOut[2] > outimagew else 0
    rt = round((shapeOut[1] - outimageh)/2) if shapeOut[1] > outimageh else 0
    iw = outimagew if shapeOut[2] > outimagew else shapeOut[2]
    ih = outimageh if shapeOut[1] > outimageh else shapeOut[1]
    
    print(outputimage.shape)

    resultoutimage = outputimage[rt:rt  + ih, lt:lt + iw, 0]
    

    resultimage = resultoutimage.astype(np.uint8)

    return resultimage




# ***********************************************************************
# Capture realtime video stream using OpenCV
# ***********************************************************************

capture = cv.VideoCapture(0)

if capture.isOpened() == False:
    print("Video stream can't be open!")
    sys.exit()


capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

key = 1
while True:
    retval, inputimage = capture.read()
    
    #preprocess the input images
    preprocessedImage, imageYCrCb = preprocess_fn(inputimage)
    
    #input the images into RDN network
    image[0, ...] = preprocessedImage

    #execute the RDN network
    job_id = dpu.execute_async(input_data, output_data)
    dpu.wait(job_id)

    #output the images from RDN network
    temp = [j.reshape(1, outputSize) for j in output_data]
    
    #postprocess the result images
    resultY = temp[0][0].reshape(shapeOut[1:])
    resultY = (resultY)*255.0
    resultY = np.clip(resultY, 0.0, 255.0)
    
    resultout = postprocess_fn(resultY, 1280, 960)
    
    print(resultout.shape)
    #set the output image to Y channel result image
    imageYCrCb[..., 0] = resultout
    #convert the YCrCb image to BGR image
    resultBGR = cv.cvtColor(imageYCrCb, cv.COLOR_YCrCb2BGR)
    
    #display the result image
    cv.imshow("Result", resultBGR)

    key = cv.waitKey(1)

    if key == ord('q'):
       break


capture.release()
cv.destroyAllWindows()





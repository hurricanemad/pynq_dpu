# Super Resolution application for Kria-RoboticsAI

## Introduction: 

This program is a sample of the [Kria-RoboticsAI](https://github.com/amd/Kria-RoboticsAI?tab=readme-ov-file). You can use these codes to run RDN(Residual Dense Network for Image Super-Resolution) and SRCNN(Residual Dense Network for Image Super-Resolution) in KR260 Kit. I provide the codes to evaluate the PSNR and SSIM of quantized model and you can use these codes to judge the performance of these super-resolution model. Besides, I provide the codes for you , which capture the realtime stream of USB camera and perform the super-resolution operation to the captured images using OpenCV toolkit. If you want to integrate the repository, you can cite  [SR code for KR260 kit](https://github.com/hurricanemad/pynq_dpu).

## Author: Hui Shen(Doxxxx)

## Date: June, 2024

## Premise:

Before using these codes, you should install the [Kria-RoboticsAI](https://github.com/amd/Kria-RoboticsAI?tab=readme-ov-file) in the KR260 kit. After the installation, PYNQ DPU depoly in the KR260 board. Then, you reboot the board and run below codes to enter the pynq environment:

 you should start as the super user

**sudo su**

then you can set the pynq environment to initalize the development

**source /etc/profile.d/pynq_venv.sh**

Additionally, after intalling the pynq_venv environment, the jupyter notebook home folder is created. You can use the below code to enter the folder:

**cd $PYNQ_JUPYTER_NOTEBOOKS**

After running the code, you would enter the path '/root/jupyter_notebooks', which is the working path.

## Quick Start:

You can clone this repository in the jupyter notebook home folder. Generally, the 'pynq-dpu' folder have existed in the jupyter notebooks folder. However, you can use the cloned repository for super-resolution test.

I have quantized two super-resolution network, which are RDN and SRCNN, repectively. You can reivew these operation code in each folder named after the method. As an example, you can use the below command to use the RDN network.

at first, you should intialze the pynq-dpu envrionment as super user

**sudo su**

**source  /etc/profile.d/pynq_venv.sh**

**cd $PYNQ_JUPYTER_NOTEBOOKS**

then, you can enter the RDN folder to evaluation the PSNR and SSIM of quantized model. Before that, I have performed quantization in host machine. You can clone [this repository](https://github.com/hurricanemad/Vitis_AI) to review the quantization code.

**cd pynq_dpu/RDN**

you can run the python command to operate the RDN network

**python3 app_MedImage_pytorch_RDN.py**

you can run the cpp exe using below command, this command can run the program compiled from the build_app.sh.
it ouput the RDN and you can use the RDN to run_cnn.exe. it can operate the images in the medicalx2df folder and evaluation the PSNR of quantized model.

**python3 app_run_RDN cpp_code.py** 


In the 'pynq-dpu' folder, there is another super-resolution network 'SRNet'. Like RDN network, you can use the below commands to run this network.

**sudo su**

**source  /etc/profile.d/pynq_venv.sh**

**cd $PYNQ_JUPYTER_NOTEBOOKS**

**cd pynq_dpu/SRNet**

you can run SRNet model using the python codes.

**python3 app_MedImage_pytorch_SRNet.py**

you can run SRNet and evaluate the PSNR of quantized model using the cpp codes.

**python3 app_run_SRCNN_cpp_code.py** 

Except for above code, I also provide a realtime super resolution program, which implements the RDN network.
you can enter the 'realtimeRDN' folder and use below code to rum this methods.

**cd $PYNQ_JUPYTER_NOTEBOOKS**

**cd pynq_dpu/realtimeRDN**

you can use the below command to start the camera capture and upscale the realtime video images

**python3 app_Camera_pytorch_RDN.py**

you can use another cpp command to start the camera capture and upscale the realtime video images

**python3 app_run_RDN_cpp_code.py**

you can also use below command to start the camera capture and run SRNet network

**cd $PYNQ_JUPYTER_NOTEBOOKS**

**cd pynq_dpu/realtimeSRNet**

run the python code using below code

**python3 app_Camera_pytorch_SRCNet.py**

run the cpp code using below code

**python3 app_run_SRCNet_cpp_code.py**

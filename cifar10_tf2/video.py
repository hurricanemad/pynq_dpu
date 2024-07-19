import cv2 as cv


img  = cv.imread("//root//jupyter_notebooks//pynq-dpu//cifar10_tf2//test_images//deer_9937.png", -1)

cv.imshow("sourceimage", img)
cv.waitKey(0)

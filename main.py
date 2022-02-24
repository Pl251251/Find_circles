import cv2
from PIL import Image
import math
import numpy as np
#pillow
im = Image.open('test_l3.jpg')
im_rotate = im.rotate(348)
im = im_rotate.save("hello.jpg")
#opencv
img = cv2.imread('hello.jpg',0)


cv2.imshow('oval', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
y = int((img.shape[1]) * (8.5/(math.sqrt(230))))
resized_ = cv2.resize(img,(y, img.shape[0]))

#finding a circle
blur= cv2.medianBlur(resized_,7)
gray = cv2.cvtColor(blur,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=100,maxRadius=120)
info = np.uint16(np.around(circles))
a=0
for i in info[0,:]:
    cv2.circle(resized_,(i[0],i[1]),i[2],(255,0,0),2)
    cv2.circle(resized_,(i[0],i[1]),2,(0,0,255),3)

img =resized_
x_2 = int((img.shape[1]) * ((math.sqrt(230)/8.5)))
resized2_ = cv2.resize(img,(x_2, img.shape[0]))



cv2.imwrite("hello4.jpg", resized2_)
#pillow
im = Image.open('hello4.jpg')
im_rotate = im.rotate(12)
im = im_rotate.save("hello.jpg")
#opencv
img = cv2.imread('hello.jpg',0)



cv2.imshow('oval', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

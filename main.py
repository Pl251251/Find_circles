import cv2
from PIL import Image
import math
import numpy as np
#pillow
im = Image.open('test_l3.jpg')

w,h =im.size
im2 = Image.new(im.mode, (2*w, 2*h), (255,255,255))
im2.paste(im, (int(w/2), int(h/2)))

im_rotate = im2.rotate(348)
im_rotate.save("test.jpg")
#opencv
img = cv2.imread('test.jpg', 1)
width = img.shape[1]
height = img.shape[0]
x_1 = int(width * (8.5/(math.sqrt(240))))
resized_ = cv2.resize(img, (x_1, height))

#finding a circle
blur= cv2.medianBlur(resized_,7)
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=100,maxRadius=120)
info = np.uint16(np.around(circles))
a=0
for i in info[0,:]:
    cv2.circle(resized_,(i[0],i[1]),i[2],(255,0,0),2)
    cv2.circle(resized_,(i[0],i[1]),2,(0,0,255),3)

resized2_ = cv2.resize(resized_,(width, height))

cv2.imwrite("test.jpg", resized2_)
#pillow
im = Image.open('test.jpg')
im_rotate = im.rotate(12)
im_rotate.save("test.jpg")

#opencv
img = cv2.imread('test.jpg',1)
dim1 = img.shape[1]
dim2 = img.shape[0]
dim3 = int((dim1/2) - w/2)
dim4 = int((dim2/2) -h/2)
dim5 = int((dim1/2) + w/2)
dim6 = int((dim2/2) +h/2)
img = img[dim4:dim6,dim3:dim5]
cv2.imwrite("final.jpg", img)

cv2.imshow('oval', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

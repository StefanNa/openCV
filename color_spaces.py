
import numpy as np
import cv2

##lesson 17 video inputs
cap = cv2.VideoCapture(CAP_OPENNI_BGR_IMAGE)

while(True):
    ret, frame = cap.read()
    
    frame= cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
    cv2.imshow("Frame",frame)
    
    
    
    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
##lesson16 - scale and rotate

img= cv2.imread("images/lego1.jpg", 1)
height,width,channels = image.shape

#scale
img_half = cv2.resize(img, (0,0),fx=0.5, fy=0.5)
#explicit scaling to 600x600
img_stretch = cv2.resize(img, (600, 600))
img_stretch_near = cv2.resize(img, (600, 600), interpolation = cv2.INTER_NEAREST)

#rotate
M= cv2.getRotationMatrix2D((0,0),-30,1)

rotated=cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))


cv2.imshow("Half", img_half)
cv2.imshow("Stretch",img_stretch)
#cv2.moveWindow("Stretch",0,2*height)
cv2.imshow("Stretch near",img_stretch_near)
#cv2.moveWindow("Stretch near",1.5*width,0.3*height)
cv2.imshow("Rotated",rotated)
#cv2.moveWindow("Rotated",3*width,2*height)

cv2.waitKey(0)
cv2.destroyAllWindows()

##lesson15 -gaussian blur, Dilation, Erosion
image= cv2.imread("images/lego1.jpg", 1)
height,width,channels = image.shape


blur=cv2.GaussianBlur(image, (5,55),0)

kernel = np.ones((8,8),"uint8")

dilate= cv2.dilate(image, kernel,iterations=1)
erode= cv2.erode(image, kernel,iterations=1)

# cv2.imshow("Original", image)
# cv2.imshow("Blur",blur)
# cv2.moveWindow("Original",0,2*height)

cv2.imshow("Dilate", dilate)
cv2.imshow("Erode",erode)
cv2.moveWindow("Erode",0,2*height)


cv2.waitKey(0)
cv2.destroyAllWindows()

##lesson 14 - transparancy layer
color= cv2.imread("images/lego1.jpg", 1)
height,width,channels = color.shape
cv2.imshow("Image", color)
cv2.moveWindow("Image",0,0)

grey= cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
cv2.imwrite("images/lego1_grey.jpg", grey)

b= color[:,:,0]
g= color[:,:,1]
r= color[:,:,2]

# make anything but green transparent
rgba=cv2.merge((b,g,r,g))
cv2.imwrite("images/lego1_rgba.png", rgba)


cv2.imshow("Image", color)
cv2.moveWindow("Image",0,0)

cv2.imshow("grey", grey)
cv2.moveWindow("grey",width,height)

cv2.imshow("rgba", rgba)
cv2.moveWindow("rgba",0,2*height)

cv2.waitKey(0)
cv2.destroyAllWindows()

##lesson13
img= cv2.imread("images/lego2.jpg", 1)
cv2.imshow("Image", img)
cv2.moveWindow("Image",0,0)
print(img.shape)
height,width,channels = img.shape

b,g,r = cv2.split(img)
b0,g0,r0=cv2.split(np.zeros([height,width,channels],"uint8"))

rgb_split_test = np.empty([height,width*3,channels],"uint8")
rgb_split_test[:,0:width] =cv2.merge([b,g0,r0])
rgb_split_test[:,width:width*2] =cv2.merge([b0,g,r0])
rgb_split_test[:,width*2:width*3] =cv2.merge([b0,g0,r])

cv2.imshow("interest", rgb_split_test)

rgb_split = np.empty([height,width*3,channels],"uint8")
rgb_split[:,0:width] =cv2.merge([b,b,b])
rgb_split[:,width:width*2] =cv2.merge([g,g,g])
rgb_split[:,width*2:width*3] =cv2.merge([r,r,r])

cv2.imshow("Channels", rgb_split)
cv2.moveWindow("Channels",0,height)

hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)
hsv_split= np.concatenate((h,s,v),axis=1)
cv2.imshow("Split HSV",hsv_split)
cv2.moveWindow("Split HSV",0,2*height)

cv2.waitKey(0)
cv2.destroyAllWindows()

##lesson 12
# black= np.zeros([150,200,1],'uint8')
# cv2.imshow("Black",black)
# print(black[0,0,:])
# 
# ones=np.ones([150,200,3],"uint8")
# cv2.imshow("Ones",ones)
# print(ones[0,0,:])
# 
# white=np.ones([150,200,3],"uint16")
# 
# white *=(2**16 -1)
# cv2.imshow("White",white)
# print(white[0,0,:])
# 
# color = ones.copy()
# color[:,:]=(255,0,0)
# cv2.imshow("Blue",color)
# print(white[0,0,:])



##lesson11
###import image, get info about image, display image and save image
#img= cv2.imread("images/lego1.jpg", 1)
#img.shape
#img.dtype
#cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
#cv2.imshow("Image", img)
#cv2.waitKey(0)
#cv2.imwrite("images/image.jpg",img)

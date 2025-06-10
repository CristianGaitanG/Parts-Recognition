import numpy as np
import cv2
import neurolab as neuro

def otsu_thresh(im):
    mini=np.amin(im)
    maxi=np.amax(im)
    
    rng = np.linspace(mini,maxi,num=128)
    bcv = np.zeros(rng.size)

    for i in range(rng.size-1):
        r1=im[im<rng[i]]
        r2=im[im>=rng[i]]

        m1 = np.mean(r1)
        m2 = np.mean(r2)
        dm2 = np.power((m1-m2),2)
        n1n2 = r1.size*r2.size
        bcv[i] =np.multiply(dm2,n1n2)
        
    ind=np.argmax(np.uint64(bcv))
    threshold = rng[ind]+0.5
    
    bw = (im>threshold)*255
    return bw
    
def filtering(im):
    kernel5=np.ones((5,5))
    kernel = np.array([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1]
    ]) / 7
    filt=cv2.filter2D(im,-1,kernel)
    test = otsu_thresh(filt)  
    return test
    
#Reading images
img1 = cv2.imread('bolt001.tif')
img2 = cv2.imread('eyeb001.tif')
img3 = cv2.imread('hook001.tif')
img4 = cv2.imread('loop001.tif')
img5 = cv2.imread('ubol001.tif')
img6 = cv2.imread('wash001.tif')
#Change space color
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)
img5 = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)
img6 = cv2.cvtColor(img6,cv2.COLOR_BGR2GRAY)
#Change uint8 to double
imgd1 = np.double(img1)/255
imgd2 = np.double(img2)/255
imgd3 = np.double(img3)/255
imgd4 = np.double(img4)/255
imgd5 = np.double(img5)/255
imgd6 = np.double(img6)/255
#Filtering images
filtered1 = filtering(imgd1)
filtered2 = filtering(imgd2)
filtered3 = filtering(imgd3)
filtered4 = filtering(imgd4)
filtered5 = filtering(imgd5)
filtered6 = filtering(imgd6)
#Change double to unit8
filtered1 = np.uint8(filtered1)
filtered2 = np.uint8(filtered2)
filtered3 = np.uint8(filtered3)
filtered4 = np.uint8(filtered4)
filtered5 = np.uint8(filtered5)
filtered6 = np.uint8(filtered6)

#cv2.imshow('Gray',img1)
#cv2.imshow('Binary',filtered1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

cont1,hi1 = cv2.findContours(filtered1,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
cont2,hi2 = cv2.findContours(filtered2,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
cont3,hi3 = cv2.findContours(filtered3,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
cont4,hi4 = cv2.findContours(filtered4,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
cont5,hi5 = cv2.findContours(filtered5,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
cont6,hi6 = cv2.findContours(filtered6,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
contours=np.array([cont1,cont2,cont3,cont4,cont5,cont6])
#cv2.drawContours(img,contours,-1,(0,255,0),3)

#cv2.imshow("Frame",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

feat=np.zeros((6,5))
i=0

for count in contours:
    area=cv2.contourArea(count[0])
    x,y,w,h = cv2.boundingRect(count[0])
    rect_area = w*h
    hull = cv2.convexHull(count[0])
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    extent = float(area)/rect_area
    print ("N.C. ",len(count)," AREA: ",area," EXTENT: ",extent, " SOLIDITY: ",solidity," HULL AREA: ",hull_area)
    feat[i] = [len(count),area,extent,solidity,hull_area]
    i=i+1

inp=feat
tar=np.array([[1,0,0,0,0,0],
               [0,1,0,0,0,0],
               [0,0,1,0,0,0],
               [0,0,0,1,0,0],
               [0,0,0,0,1,0],
               [0,0,0,0,0,1]])
             
net = neuro.net.newff(neuro.tool.minmax(inp),[15,6])
error = net.train(inp,tar,epochs=1500,show=100,goal=0.02)
net.save('trainedNet.net')

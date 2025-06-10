import numpy as np
import neurolab as neuro
import cv2

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
    #kernel51=np.ones((5,5),np.float32)/25
    #kernel5=np.ones((5,5))
    #kernel3=np.ones((3,3))
    kernel = np.array([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1]
    ]) / 7
    filt=cv2.filter2D(im,-1,kernel)
    test = otsu_thresh(filt)  
    return test


img = cv2.imread('hook001.tif')
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgd1 = np.double(img1)/255
filtered1 = filtering(imgd1)
filtered1 = np.uint8(filtered1)

cv2.imshow('Gray',img)
cv2.imshow('Binary',filtered1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cont1,hi1 = cv2.findContours(filtered1,cv2.CHAIN_APPROX_SIMPLE,cv2.RETR_TREE)
cv2.drawContours(img,cont1,-1,(0,255,0),3)
cv2.imshow("Frame",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

area=cv2.contourArea(cont1[0])
x,y,w,h = cv2.boundingRect(cont1[0])
rect_area = w*h
hull = cv2.convexHull(cont1[0])
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
extent = float(area)/rect_area
print ("N.C. ",len(cont1)," AREA: ",area," EXTENT: ",extent, " SOLIDITY: ",solidity," HULL AREA: ",hull_area)
feat = [len(cont1),area,extent,solidity,hull_area]


net2=neuro.load('trainedNet.net')
y3=[feat]
label=['Bolt','Eye Bolt','Hook','Loop','U-Bolt','Wash']
out = net2.sim(y3)
index=np.argmax(out)
print (out)
print ('The element is ',label[index])

#Made by Guining Pertin

import numpy as np
import cv2

#Take input using camera
cap = cv2.VideoCapture(0)

def nothing(x):
    pass

#Creating Trackbar Window
hsv = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('HSV')

#Creating Trackbars
cv2.createTrackbar('H-low', 'HSV', 0, 180, nothing)
cv2.createTrackbar('S-low', 'HSV', 20, 255, nothing)
cv2.createTrackbar('V-low', 'HSV', 50, 255, nothing)
cv2.createTrackbar('H-high', 'HSV', 30, 180, nothing)
cv2.createTrackbar('S-high', 'HSV', 150, 255, nothing)
cv2.createTrackbar('V-high', 'HSV', 255, 255, nothing)

while(1):
    ret, frame = cap.read()

    #Convertto HSV Format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Get values from Trackbars
    h1 = cv2.getTrackbarPos('H-low', 'HSV')
    s1 = cv2.getTrackbarPos('S-low', 'HSV')
    v1 = cv2.getTrackbarPos('V-low', 'HSV')
    h2 = cv2.getTrackbarPos('H-high', 'HSV')
    s2 = cv2.getTrackbarPos('S-high', 'HSV')
    v2 = cv2.getTrackbarPos('V-high', 'HSV')
    
    #Take lower and upper limits and make the mask
    skin_lower = np.array([h1,s1,v1])
    skin_upper = np.array([h2,s2,v2])
    mask = cv2.inRange(hsv, skin_lower, skin_upper)

    #Morphological Transformations and the kernels
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    opening1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
    blur1 = cv2.medianBlur(opening1, 5)
    kernel2 = np.ones((5,5),np.uint8)
    erode1 = cv2.erode(blur1, kernel2, iterations = 2)
    closing1 = cv2.morphologyEx(blur1, cv2.MORPH_CLOSE, kernel2)
    blur2 = cv2.medianBlur(closing1, 5)
    closing2 = cv2.morphologyEx(blur2, cv2.MORPH_CLOSE, kernel1)

    #Find contours
    findContour, contours, hierarchy = cv2.findContours(closing2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Find the maximum contour area
    max_area=100
    ci=0	
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
            
    #Set cnts as the largest contour
    cnts = contours[ci]

    #Find convex hull and defects
    hull = cv2.convexHull(cnts) #For the coordinates
    hull2 = cv2.convexHull(cnts, returnPoints = False)  #For the indices
    defects = cv2.convexityDefects(cnts, hull2)

    #Draw defect points
    fardefects = []
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnts[s][0])
        end = tuple(cnts[e][0])
        far = tuple(cnts[f][0])
        fardefects.append(far)  #append the far values to fardefects
        cv2.line(frame,start,end,[0,255,0],2)

    #Find the moments and the centroid
    M = cv2.moments(cnts)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    centroid = (cx,cy+30)
    
    #Draw a circle around the centroid
    cv2.circle(frame, centroid,70, [255,0,0], 2)

    #Get fingertips
    finger = []
    for i in range(0,len(hull)-1):
        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
            if hull[i][0][1] <500 :
                finger.append(hull[i][0])

    #Largest 5 hull points are the fingers
    finger =  sorted(finger,key=lambda x: x[1])   
    fingers = finger[0:5]                

    #Finger Distance from centoid
    fingerDist = []
    for i in range(0,len(fingers)):
        distance = np.sqrt(np.power(fingers[i][0]-centroid[0],2)+np.power(fingers[i][1]-centroid[0],2))
        fingerDist.append(distance)

    #Find the mean of defect distance from centroid    
    defect_centroid = []
    for i in range(0,len(fardefects)):
        x =  np.array(fardefects[i])
        centroid = np.array(centroid)
        dist = np.sqrt(np.power(x[0]-centroid[0],2)+np.power(x[1]-centroid[1],2))
        defect_centroid.append(dist)

    #Find the mean of the defect_centroid distances using largest 3 values
    mean_distance = np.mean((sorted(defect_centroid))[0:2])

        
    #Find number of fingers raised
    result = 0
    for i in range(0,len(fingers)):
        if fingerDist[i] > mean_distance+100:
            result = result +1

    print result
    
    #Show the final formats
    cv2.imshow('1',closing2)
    cv2.imshow('2',frame)

    #Wait for key to be pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break     
   
    
cap.release()
cv2.destroyAllWindows()


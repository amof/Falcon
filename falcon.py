#!/usr/bin/env python

import cv2
import numpy as np
import time

def pointToLongerDistance(cap):
    """
    Function to rotate the mobile robot to the longest safest distance detected
    Keyword arguments:
    cap -- video camera interface
    """

    # Take each frame
    _, frame = cap.read()

    # Acquisition
    # Convert to grayscale
    img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply filter
    bilateralFilter = cv2.bilateralFilter(img2gray, 9, 30, 30)
    # Edges detector
    edges = cv2.Canny(bilateralFilter, 25, 50)

    # Method with manual looking
    stepsize=4
    edgearray=[]
    imagewidth=edges.shape[1]-1
    imageheight=edges.shape[0]-1
    for j in range (0,imagewidth,stepsize):
        for i in range(imageheight-4,0,-1):
            if edges.item(i,j) == 255:
                edgearray.append((j,i))
                break
        else:
            edgearray.append((j,0))
    for x in range (len(edgearray)-1):
        cv2.line(frame, edgearray[x], edgearray[x+1],(0,255,0),1)

    for x in range (len(edgearray)):
        cv2.line(frame, (x*stepsize,imageheight), edgearray[x],(0,255,0),1)

    # Draw a line with max height and print text
    edgearrayNp = np.array(edgearray, dtype=int)
    itemindex = edgearrayNp.argmin(axis=0)[1]
    cv2.line(frame, (edgearrayNp[itemindex][0], 480), (edgearrayNp[itemindex][0], edgearrayNp[itemindex][1]), (255, 0, 0), 5)
    cv2.line(frame, (320, 480-int((480-edgearrayNp[itemindex][1])/2)), (edgearrayNp[itemindex][0], 480-int((480-edgearrayNp[itemindex][1])/2)), (0, 0, 255), 5)
    if edgearrayNp[itemindex][0] <320 :
        cv2.putText(frame, 'Left!', (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else :
        cv2.putText(frame, 'Right!', (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Draw a box and print Obstacle if a distance is shorter to the box Y value
    obstacleDetected=0
    boxObstacle_X1=150
    boxObstacle_X2 = 490
    boxObstacle_Y=430 # 480 - 50 px
    for x in range(len(edgearray)):
        if edgearray[x][1] > boxObstacle_Y and edgearray[x][0] > boxObstacle_X1 and edgearray[x][0] < boxObstacle_X2:
            cv2.line(frame, (boxObstacle_X1, boxObstacle_Y), (boxObstacle_X2, boxObstacle_Y),(0, 0, 255), 2)
            cv2.line(frame, (boxObstacle_X1, 480), (boxObstacle_X1, boxObstacle_Y), (0, 0, 255), 2)
            cv2.line(frame, (boxObstacle_X2, 480), (boxObstacle_X2, boxObstacle_Y), (0, 0, 255), 2)
            cv2.putText(frame, 'OBSTACLE', (240, 465), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            obstacleDetected=1
            break

    if obstacleDetected == 0:
        cv2.line(frame, (boxObstacle_X1, boxObstacle_Y), (boxObstacle_X2, boxObstacle_Y), (0, 255, 0), 2)
        cv2.line(frame, (boxObstacle_X1, 480), (boxObstacle_X1, boxObstacle_Y), (0, 255, 0), 2)
        cv2.line(frame, (boxObstacle_X2, 480), (boxObstacle_X2, boxObstacle_Y), (0, 255, 0), 2)

    # Print pictures
    cv2.imshow('Video Input', frame)
    #cv2.imshow('img2gray', img2gray)
    #cv2.imshow('bilateralFilter', bilateralFilter)
    #cv2.imshow('Edges', edges)


def findObject(cap):
    """
    Function to detect objects
    Keyword arguments:
    cap -- video camera interface
    """
    # Take each frame
    _, frame = cap.read()

    # Convert to grayscale
    img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bilateralFilter = cv2.bilateralFilter(img2gray, 9, 30, 30)

    # Method with built-in method findContours
    findContours = frame.copy()
    ret, thresh = cv2.threshold(bilateralFilter, 150, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 100:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            if len(approx) == 5:
                cv2.drawContours(findContours, [cnt], 0, 255, -1)
            elif len(approx) == 3:
                cv2.drawContours(findContours, [cnt], 0, (0, 255, 0), -1)
            elif len(approx) == 4:
                cv2.drawContours(findContours, [cnt], 0, (0, 0, 255), -1)
            else:
                cv2.drawContours(findContours, [cnt], 0, (255, 0, 255), -1)

    # Print pictures
    cv2.imshow('findContours', findContours)


if __name__ == "__main__":
    cv2.setNumThreads(4)
    cap = cv2.VideoCapture(0)

    numberOfFrames = 0

    while(1):
        if numberOfFrames == 0:
            startT1 = time.time()

        pointToLongerDistance(cap)
        findObject(cap)

        numberOfFrames = numberOfFrames + 1

        if numberOfFrames > 80:
            endT1 = time.time()
            seconds = endT1 - startT1
            fps = numberOfFrames / seconds
            print("FPS statistics : ", fps)
            numberOfFrames = 0

        k = cv2.waitKey(5) & 0xFF
        if k == 27: # OUT !
            break

        #time.sleep(0.001) # let the processor rest

    cv2.destroyAllWindows()
    cap.release()
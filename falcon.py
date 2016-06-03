#!/usr/bin/env python

import cv2
import numpy as np


def pointToLongerDistance(cap):
    """
    Function to rotate the mobile robot to the longest safest distance detected
    Keyword arguments:
    cap -- video camera interface
    """

    # Take each frame
    _, frame = cap.read()

    # Acquisition
    # Convert to gray
    img2gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply filter
    bilateralFilter = cv2.bilateralFilter(img2gray, 9, 30, 30)
    # Edges detector
    edges = cv2.Canny(bilateralFilter, 25, 50)

    # Method with built-in method findContours
    findContours = frame.copy()
    ret, thresh = cv2.threshold(img2gray, 150, 255, 0)
    im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(findContours, contours, -1, (0, 255, 0), 3)


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

    # Print pictures
    cv2.imshow('Video Input', frame)
    cv2.imshow('img2gray', img2gray)
    cv2.imshow('bilateralFilter', bilateralFilter)
    cv2.imshow('Edges', edges)
    cv2.imshow('findContours', findContours)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while(1):
        pointToLongerDistance(cap)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
import numpy as np
import cv2

def nmsFilter(rectangles, threshold):
    rectangles = np.asarray(rectangles)

    leftX = rectangles[:,0]
    leftY = rectangles[:,1]
    rightX = rectangles[:,2]
    rightY = rectangles[:,3]

    areas = (rightX -leftX + 1) * (rightY - leftY + 1)

    # indices of rectangles which are sorted by y-location of bottom right corner
    leftToRightRectangleIndices = np.argsort(rightY)

    correct = []
    while leftToRightRectangleIndices != []:
        last = len(leftToRightRectangleIndices) - 1
        lastRectangleIdx = leftToRightRectangleIndices[last]
        correct.append(lastRectangleIdx)
        drop = [last]
        # remove rectangles which overlap too much with the rightmost one
        for prev in range(0, last):
            prevRectangleIdx = leftToRightRectangleIndices[prev]

            intersectLeftX = Math.max(leftX[prevRectangleIdx], leftX[lastRectangleIdx])
            intersectLeftY = Math.max(leftY[prevRectangleIdx], leftY[lastRectangleIdx])
            intersectRightX = Math.min(rightX[prevRectangleIdx, rightX[lastRectangleIdx])
            intersectRightY = Math.min(rightY[prevRectangleIdx], rightY[lastRectangleIdx])

            intersectArea = max(0, intersectRightX - intersectLeftX + 1) \
                            * max(0, intersectRightY - intersectLeftY + 1)
            
            ratio = float(intersectArea) / areas[prevRectangleIdx]
            if(ratio > threshold):
                drop.append(prev)
        leftToRightRectangleIndices = np.delete(leftToRightRectangleIndices, drop)

    return rectangles[correct]


    
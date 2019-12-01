import cv2
import numpy as np
import os
import sys

 
hog = cv2.HOGDescriptor()
hog.load('myHogDector.bin')


def intersection(rect1, rect2):
    intersectLeftX = max(rect1[0], rect2[0])
    intersectLeftY = max(rect1[1], rect2[1])
    intersectRightX = min(rect1[2], rect2[2])
    intersectRightY = min(rect1[3], rect2[3])

    intersectArea = max(0, intersectRightX - intersectLeftX + 1) \
                    * max(0, intersectRightY - intersectLeftY + 1)
    
    return intersectArea


def nmsFilter(rectangles, threshold):
    print("Number of rectangles:", len(rectangles))
    if(len(rectangles) == 0):
        return []
    rectangles = np.asarray(rectangles)

    leftX = rectangles[:,0]
    leftY = rectangles[:,1]
    rightX = rectangles[:,2]
    rightY = rectangles[:,3]

    areas = (rightX -leftX + 1) * (rightY - leftY + 1)

    # indices of rectangles which are sorted by y-location of bottom right corner
    leftToRightRectangleIndices = np.argsort(rightY)

    # print("Original indices", leftToRightRectangleIndices)

    correct = []
    while leftToRightRectangleIndices != []:
        print("We still have ", leftToRightRectangleIndices)
        last = len(leftToRightRectangleIndices) - 1
        lastRectangleIdx = leftToRightRectangleIndices[last]
        correct.append(lastRectangleIdx)
        drop = [last]
        # remove rectangles which overlap too much with the rightmost one
        for prev in range(1, last):
            prevRectangleIdx = leftToRightRectangleIndices[prev]

            # intersectLeftX = max(leftX[prevRectangleIdx], leftX[lastRectangleIdx])
            # intersectLeftY = max(leftY[prevRectangleIdx], leftY[lastRectangleIdx])
            # intersectRightX = min(rightX[prevRectangleIdx], rightX[lastRectangleIdx])
            # intersectRightY = min(rightY[prevRectangleIdx], rightY[lastRectangleIdx])

            # intersectArea = max(0, intersectRightX - intersectLeftX + 1) \
            #                 * max(0, intersectRightY - intersectLeftY + 1)

            intersectArea = intersection(rectangles[prevRectangleIdx], rectangles[lastRectangleIdx])
            
            ratio = float(intersectArea) / areas[prevRectangleIdx]
            if(ratio > threshold):
                drop.append(prev)

        leftToRightRectangleIndices = np.delete(leftToRightRectangleIndices, drop)

    return rectangles[correct].tolist()


def mergeRects(rectangles):
    if(rectangles == []):
        return []
    rectangles = np.asarray(rectangles)
    leftX, leftY, rightX, rightY = float("inf"), float("inf"), -1, -1
    merged = [rectangles[0]]
    for rect in rectangles[1:]:
        found = False
        for i in range(len(merged)):
            m = merged[i]
            if intersection(m, rect) > 0:
                newRect = [min(m[0], rect[0]), min(m[1], rect[1]), max(m[2], rect[2]), max(m[3], rect[3])]
                merged[i] = newRect
                found = True
                # break
        if not found:
            merged.append(rect)
    # leftX = min(rectangles[:,0])
    # leftY = min(rectangles[:,1])
    # rightX = max(rectangles[:,2])
    # rightY = max(rectangles[:,3])
    return merged


def predict(img, modelPath, merge=False):
    hog = cv2.HOGDescriptor()
    hog.load(modelPath)
    rects, wei = hog.detectMultiScale(img, winStride = (4, 4), padding = (8, 8), scale=1.05)
    # print(rects)
    locations = []
    for (x, y, H, W) in rects:
        locations.append([x, y, x+H, y+W])
    newRects = nmsFilter(locations, 0.3)
    # newRects = locations
    if merge:
        return mergeRects(newRects)
    else:
        return newRects


def test(dirPath, modelPath, num, merge = False):
    files = os.listdir(dirPath)
    num = min(len(files), num)
    i = 0
    for f in files:
        img = cv2.imread(dirPath + f)
        # rects, wei = hog.detectMultiScale(img, winStride = (4, 4), padding = (8, 8), scale=1.05)
        # # print(rects)
        # locations = []
        # for (x, y, H, W) in rects:
        #     locations.append([x, y, x+H, y+W])
        # newRects = nmsFilter(locations, 0.3)
        # # newRects = locations
        # merged = mergeRects(newRects)
        rects = predict(img, modelPath, merge)
        # if merge:
        #     for (x1, y1, x2, y2) in merged:
        #         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # else:
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        i += 1
        cv2.imshow("test_" + str(i), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("test_" + modelPath + "-" + str(i) + ".jpg", img)
        if(num == i):
            break


if __name__ == "__main__":
    parameters = sys.argv
    dirPath = parameters[1]
    modelPath = parameters[2]
    number = int(parameters[3])
    print(number)
    merge = bool(parameters[4])
    test(dirPath, modelPath, number, merge=True)
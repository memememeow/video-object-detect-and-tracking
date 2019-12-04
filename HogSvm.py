import os
import sys
import cv2
import numpy as np

def computeHOGs(images, size=(128, 64)):
    gradientList = []
    hog = cv2.HOGDescriptor()
    for img in images:
        # compute HOG descriptor for one image
        if img.shape[1] > size[1] and img.shape[0] > size[0]:
            # if the image is too big, we crop the central part of the image
            H, W = img.shape[:2]
            roi = img[(H - size[0]) // 2: (H - size[0]) // 2 + size[0], (W - size[1]) // 2: (W - size[1]) // 2 + size[1]]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = np.power(img/float(np.max(gray)), 1.5)
            gradientList.append(hog.compute(gray))
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gamma correction
            gray = (np.power(gray/float(np.max(gray)), 1/1.5) * 255).astype(np.uint8)
            gradientList.append(hog.compute(gray))
    return gradientList
    
 
def createSvm():
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    # set soft classifier
    svm.setC(0.01)
    # do regression
    svm.setType(cv2.ml.SVM_EPS_SVR)
    return svm
 
def getSvmDetector(svm):
    vectors = svm.getSupportVectors()
    vectors = np.transpose(vectors)
    rho = svm.getDecisionFunction(0)[0]
    return np.append(vectors, [[-rho]], 0)

def findHardExamples(svm, negList):
    hardExamples = []
    hog = cv2.HOGDescriptor()
    detector = getSvmDetector(svm)
    hog.setSVMDetector(detector)
    # let hog detect each negative example, hog is expected to give no rectangles when given negative examples
    for neg in negList:
        rects, weights = hog.detectMultiScale(neg, winStride=(4, 4),padding=(8, 8), scale=1.05)
        # crop out each rectangle part which misleads the model and resize them to be a new negative example
        for (x,y,H,W) in rects:
            hardExample = neg[y:y+W, x:x+H]
            hardExamples.append(cv2.resize(hardExample,(64, 128)))
    return hardExamples       

def loadImages(dirPath, num = 30):
    imges = []
    # dirname is path to the direcotry where images are stored
    files = os.listdir(dirPath)
    for f in files:
        fileName = dirPath + "/" + f
        imges.append(cv2.imread(fileName))
        num -= 1
        # if amount limit has been reached, we stop and ignore how many images are left
        if num <= 0:
            break
    return imges

if __name__ == "__main__":
    posPath = sys.argv[1]
    negPath = sys.argv[2]
    modelPath = sys.argv[3]

    gradientList = []
    labels = []

    # for now, pos-50 and neg-110 have the best performance
    positiveList = loadImages(posPath, num = 100)
    negativeList = loadImages(negPath, num = 200)
    # sample_neg(full_neg_lst, negativeList, [128, 64])

    print("The number of positive examples:", len(positiveList))
    print("The number of negative examples:", len(negativeList))

    # gradienList stores gradients of both positive and negative examples
    # labels stores 1 for positive examples and -1 for engative examples
    gradientList = computeHOGs(positiveList) + computeHOGs(negativeList)
    labels = [1 for _ in positiveList] + [-1 for _ in negativeList]
    
    # train svm on positive and negative examples
    # training svm needs images' gradients and their corresponding labels
    svm = createSvm()
    svm.train(np.array(gradientList), cv2.ml.ROW_SAMPLE, np.array(labels))
    print("Finished first round of training")
    
    # search for negative examples which cause poor performance
    # negative examples which cause model to give rectangles are hard
    hardNegList = findHardExamples(svm, negativeList)
    print("Size of hard_negativeList:", len(hardNegList))
    gradientList += computeHOGs(hardNegList)

    # extend labels by adding -1 for new negative examples
    for i in range(len(hardNegList)):
        labels.append(-1)

    svm.train(np.array(gradientList), cv2.ml.ROW_SAMPLE, np.array(labels))
    print("Finished second round of training")
    
    # comebine hog with our trained svm detector, then save svm+hog model
    detector = getSvmDetector(svm)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(detector)
    hog.save(str(len(positiveList)) + "-" + str(len(negativeList)) + "-" + modelPath)
    print("Saved!")
 
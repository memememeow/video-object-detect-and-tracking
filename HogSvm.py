import cv2
import numpy as np
import random
import os
import sys
 
def load_images(dirPath, num = 30):
    img_list = []
    # dirname is path to the direcotry where images are stored
    files = os.listdir(dirPath)
    print(dirPath)
    print("files:", len(files))
    for f in files:
        img_name = dirPath + "/" + f
        # print(img_name)
        img_list.append(cv2.imread(img_name))
        num -= 1
        # if amount limit has been reached, we stop and ignore how many images are left
        if num <= 0:
            break
    # print(len(img_list))
    return img_list
 

def computeHOGs(img_lst, gradient_lst, wsize=(128, 64)):
    hog = cv2.HOGDescriptor()
    print(len(img_lst))
    for img in img_lst:
        if img.shape[1] > wsize[1] and img.shape[0] > wsize[0]:
            H, W = img.shape[:2]
            roi = img[(H - wsize[0]) // 2: (H - wsize[0]) // 2 + wsize[0], (W - wsize[1]) // 2: (W - wsize[1]) // 2 + wsize[1]]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gradient_lst.append(hog.compute(gray))
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gradient_lst.append(hog.compute(gray))
    return gradient_lst
    
 
 
def get_svm_detector(svm):
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)
 

if __name__ == "__main__":
    posPath = sys.argv[1]
    negPath = sys.argv[2]
    modelPath = sys.argv[3]

    neg_list = []
    pos_list = []

    gradient_lst = []
    labels = []
    hard_neg_list = []

    # for now, pos-50 and neg-110 have the best performance
    pos_list = load_images(posPath, num = 100)
    neg_list = load_images(negPath, num = 200)
    # sample_neg(full_neg_lst, neg_list, [128, 64])

    print("The number of positive examples:", len(pos_list))
    print("The number of negative examples:", len(neg_list))

    computeHOGs(pos_list, gradient_lst)
    computeHOGs(neg_list, gradient_lst)
    # print("Size of gradient list", len(gradient_lst))
    labels = [1 for _ in range(len(pos_list))] + [-1 for _ in range(len(neg_list))]
    
    # train svm on positive examples
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv2.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task
    svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))
    print("Finished first round of training")
    
    # train svm on negative examples which cause poor performance
    hog = cv2.HOGDescriptor()
    hard_neg_list.clear()
    hog.setSVMDetector(get_svm_detector(svm))
    for i in range(len(neg_list)):
        rects, wei = hog.detectMultiScale(neg_list[i], winStride=(4, 4),padding=(8, 8), scale=1.05)
        for (x,y,w,h) in rects:
            print(x, y, w, h)
            hardExample = neg_list[i][y:y+h, x:x+w]
            hard_neg_list.append(cv2.resize(hardExample,(64, 128)))
    print("Size of hard_neg_list:", len(hard_neg_list))
    computeHOGs(hard_neg_list, gradient_lst)

    for i in range(len(hard_neg_list)):
        labels.append(-1)

    # print("Number of labels", len(labels))
    svm.train(np.array(gradient_lst), cv2.ml.ROW_SAMPLE, np.array(labels))
    print("Finished second round of training")
    
    
    # save svm+hog model
    hog.setSVMDetector(get_svm_detector(svm))
    hog.save(modelPath)
    print("Saved!")
 


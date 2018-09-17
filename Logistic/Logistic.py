# coding=utf-8
import numpy as np
def loadData():
    dataSet = []
    trainSet = []
    labeltrainMat = []
    testSet = []
    labeltestMat = []
    counter = 0
    with open('./abalone.data') as fr:
        thedata = fr.readlines()
        matlen = len(thedata)
        print(matlen)
        for line in thedata:
            theline = line.strip().split(',')
            lineArr = np.array(theline[1:3], dtype=np.float)
            if theline[0] != 'I' and counter < matlen * 0.8 and len(lineArr) == 2:
                trainSet.append(lineArr)
                if (theline[0] > 'F'):
                    labeltrainMat.append(1)
                else:
                    labeltrainMat.append(0)
            elif theline[0] != 'I' and len(lineArr) == 2:
                testSet.append(lineArr)
                if (theline[0] > 'F'):
                    labeltestMat.append(1)
                else:
                    labeltestMat.append(0)
            counter += 1
        return  trainSet, labeltrainMat, testSet, labeltestMat

def sigmoid(inx):
    return 1.0 / (1 + np.exp(-inx))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    intercept = np.ones((dataMatrix.shape[0], 1))
    features = np.hstack((intercept, dataMatrix))
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(features)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(np.dot(features, weights))
        error = (labelMat - h)
        gradAscent = np.dot(features.transpose(), error)
        weights = weights + alpha * gradAscent
    return weights

def getRate(weights, testMatIn, testLabels):
    testLabels = np.mat(testLabels)
    errornum = 0
    testMatrix = np.mat(testMatIn)
    intercept = np.ones((testMatrix.shape[0], 1))
    features = np.hstack((intercept, testMatrix))
    predict = np.dot(features, weights)
    predict = predict.transpose()
    print(predict)
    testLen = len(testLabels)
    for i in range(testLen):
        if testLabels[0, i] != predict[0, i]:
            errornum += 1
    rate = errornum / testLen
    return rate

if __name__ == "__main__":
    trainSet, labeltrainMat, testSet, labeltestMat = loadData()
    weight = gradAscent(trainSet, labeltrainMat)
    print(weight)
    rate = getRate(weight, testSet, labeltestMat)
    print(rate)


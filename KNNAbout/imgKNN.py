# coding=utf-8
import re

from KNNAbout import KNN
from numpy import *
from os import listdir

def imgVector(filename):
    imgMatrix = zeros((1, 1024))
    with open(filename, 'r') as f:
        for i in range(32):
            line = f.readline()
            line = line.strip()
            for j in range(len(line)):
                imgMatrix[0, 32*i+j] = int(line[j])
    return imgMatrix

def handWritingTest(trainfilepath, testdatafilepath):
    trainingFileList = listdir(path=trainfilepath)
    hwLabels = []
    m = len(trainingFileList)
    imgMatrixs = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileName = fileNameStr.split(".")[0]
        classfileName = int(fileName.split("_")[0])
        hwLabels.append(classfileName)
        imgMatrixs[i,:] = imgVector(trainfilepath + "\\" + fileNameStr)
    testFileList = listdir(testdatafilepath)
    errorCount = 0.0
    for j in range(len(testFileList)):
        testFileNameStr = testFileList[j]
        testFileName = testFileNameStr.split(".")[0]
        testClassFileName = int(testFileName.split("_")[0])
        theTestImgMatrix = imgVector(testdatafilepath + "\\" + testFileList[j])
        result = KNN.classify0(theTestImgMatrix, hwLabels, imgMatrixs, 3)
        if (result != testClassFileName): errorCount += 1.0
        print("the recognition number is %d, the real number is %d" % (result, testClassFileName))

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/(len(testFileList))))



def findNumName(filepath):
    pattern = re.compile('\\\\(\d+)\_(\d+)\.')
    result =  re.search(pattern, filepath)
    return result.group(1), result.group(2)

#def dealAbsolutePath(filename):
    #pattern = re.compile('[a-zA-Z]+:\\\\.*?')
    #result = re.search(pattern, filename)
    #if(result != None):
        #filename.replace("\\", "\\"+"\\")
        #print(filename)
    #return filename

if __name__ == "__main__":
    trainfilepath = "C:\\Users\\dell\\Desktop\\meachineLearningPractice\\machinelearninginaction\\Ch02\\trainingDigits"
    testdatafilepath = "C:\\Users\\dell\\Desktop\\meachineLearningPractice\\machinelearninginaction\\Ch02\\testDigits"
    handWritingTest(trainfilepath, testdatafilepath)
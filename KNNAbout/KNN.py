# coding=utf-8
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createKNN():
    group = array([[2,3],[2,3.5],[10,6],[10,8]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
#算法核心 将输入的向量和每一个已知的向量相减，算出获得向量的长度，即距离
#并按照距离长度来进行排序，观察最近的前K条，前K条中，出现的最多的那种向量，即该输入向量的类型
def classify0(inX, classLabels, classGroup, k):
    datasetSize = classGroup.shape[0]
    diffMat = tile(inX, (datasetSize, 1)) - classGroup
    sqDiffMat = diffMat ** 2#每个元素平方
    sqDistances = sqDiffMat.sum(axis=1)#将每列相加
    distances = sqDistances ** 0.5#每个元素开方
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = classLabels[sortedDistIndicies[i]]#voteIlabel为整型
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]#, classCount

#将txt文件里面一定格式的数据，转化成算法可以识别的向量矩阵
def fileMatrix():
    with open(r"C:\Users\dell\Desktop\meachineLearningPractice\machinelearninginaction\Ch02\datingTestSet2.txt", 'r') as f:
        arrayLine = f.readlines() #有时候文件过大需要做一定得操作
        #readline 一次性只读取一行 是一个yield迭代器 而readlines一次性读取整个文件
        lineNum = len(arrayLine)
        returnMat = zeros((lineNum, 3))
    classLabelVector = []
    index = 0
    for line in arrayLine:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))#储存为整型
        index += 1

    return returnMat, classLabelVector

#matplotlib 绘图测试
def plotlib(datingDataMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    plt.show()

def numToLike(num):
    if num == 3:
        print("LargeDoes")
    elif num == 2:
        print("smallDoes")
    elif num == 1:
        print("dontLike")

#归一化数据 预防向量中某个分量影响过于严重
def autoNorm(dataSet):
    maxVal = dataSet.max(0)
    minVal = dataSet.min(0)
    ranges = maxVal - minVal
    normDataMat = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataMat = dataSet - tile(minVal, (m, 1))
    normDataMat = normDataMat / tile(ranges, (m, 1))
    return normDataMat, minVal, ranges

#算法测试
def datingSetMatrixTest():
    hoRatio = 0.1
    datingMatrix, labelsMatrix = fileMatrix()
    normMat, minVal, ranges= autoNorm(datingMatrix)
    m = datingMatrix.shape[0]
    testNum = int(m * hoRatio)
    errorNum = 0.0
    for i in range(testNum):
        result = classify0(normMat[i,:], labelsMatrix[testNum:m], normMat[testNum:m, :], 3)
        print("No.%d, the predict result is %s, the real result is %s" % (i, result, labelsMatrix[i]))
        if (result != labelsMatrix[i]):
            errorNum += 1.0
    print("the total error rate is: %f" % (errorNum/testNum))


if __name__ == "__main__":
    group, labels = createKNN()
    #print(classify0([7, 6], labels, group, 3))

    #dataMat, dataLabelVector = fileMatrix()
    #plotlib(dataMat)

    #numToLike(classify0([250000, 2.9, 0.8], dataLabelVector, dataMat, 100))
    #a, b = classify0([250000, 2.9, 0.8], dataLabelVector, dataMat, 100)
    #print((5).__class__.__name__)
    #print(a)
    #print(b)
    #print(a.__class__.__name__)
    #print(b.__class__.__name__)

    datingSetMatrixTest()
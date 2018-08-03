# coding=utf-8
import pickle
from math import log
import operator

#计算香农熵(该值反映的是信息的无序度)
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        labelCounts[currentLabel] = labelCounts.get(currentLabel, 0) + 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#创建一组数据:
def createDataSet():
    #dataSet最后一列表示的是类型
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers', 'fish']
    return dataSet, labels

#按照特征划分数据，axis为特征所在的列，value为该特征取的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return  retDataSet

#选择最好的数据集划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            #一个分类产生的所有结果的香农熵按照权重求得的期望，为该种分类的数据种类复杂度
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > 0):
            baseEntropy = newEntropy
            bestFeature = i
    #返回的是最好的分类特征的索引
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDecisionTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    #classList中和第一项的值相同的项的数量等于classList的长度 证明这个List里面项全部都是一样的
    #即类别完全相同
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    #dataSet某一行向量中只有一个元素，证明其用于分类的特征已经用完
    #此时返回出现的次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createDecisionTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    #secondSplit为第一个分叉点的两个不同的取值对应的树
    secondSplit = inputTree[firstStr]
    #第一个分类的特征位于向量中的第几维度
    featIndex = featLabels.index(firstStr)
    for key in secondSplit.keys():
        if testVec[featIndex] == key:
            if type(secondSplit[key]).__name__ == 'dict':
                classLabels = classify(secondSplit[key], featLabels, testVec)
            else: classLabels = secondSplit[key]
    return classLabels

def storeTree(inputTree , filename):
    with open(filename, 'w') as f:
        pickle.dump(inputTree, f)

def grabTree(filename):
    with open(filename, 'r') as f:
        return pickle.load(f)

if __name__ == "__main__":
    with open('C:\\Users\\dell\\Desktop\\meachineLearningPractice\\machinelearninginaction\\Ch03\\lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
        theTree = createDecisionTree(lenses, lensesLabels)
    print(theTree)
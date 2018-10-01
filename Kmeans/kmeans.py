import numpy as np
import math as mt
import random

def calEuclideanDistance(va, vb):
    dist = np.sqrt(np.sum(np.square(va - vb)))
    return dist

def calMin(a, b, c):
    min = 0
    if a <= b:
        if a <= c:
           min = a
    elif b <= c:
        min = b
    else:
        min = c
    return min

#三类分类kmeans
class kmeans():
    def __init__(self, points):
        self.__points = np.array(points)
        self.__classCenterA, self.__classCenterB, self.__classCenterC = self.randomChoose()
        self.__shape = np.shape(self.__points)
        self.__classA = []
        self.__classB = []
        self.__classC = []

    def initClassArray(self):
        self.__classA = []
        self.__classB = []
        self.__classC = []

    def randomChoose(self):
        na = mt.floor(random.random())
        nb = mt.floor(random.random())
        nc = mt.floor(random.random())
        va = self.__points[na]
        vb = self.__points[nb]
        vc = self.__points[nc]
        return va, vb, vc

    def classficationPoint(self):
        len = self.__shape[0]
        self.initClassArray()
        for i in range(len):
            da = calEuclideanDistance(self.__points[i], self.__classCenterA)
            db = calEuclideanDistance(self.__points[i], self.__classCenterB)
            dc = calEuclideanDistance(self.__points[i], self.__classCenterC)
            min = calMin(da, db, dc)
            if min == da:
                self.__classA.append(self.__points[i])
            if min == db:
                self.__classB.append(self.__points[i])
            if min == dc:
                self.__classC.append(self.__points[i])

    def updatePointC(self):
        pointsA = np.array(self.__classA)
        pointsB = np.array(self.__classB)
        pointsC = np.array(self.__classC)
        newCenterA = np.ceil(np.sum(pointsA, axis=0) / (np.shape(pointsA)[0]))
        newCenterB = np.ceil(np.sum(pointsB, axis=0) / (np.shape(pointsB)[0]))
        newCenterC = np.ceil(np.sum(pointsC, axis=0) / (np.shape(pointsC)[0]))
        self.__classCenterA = newCenterA
        self.__classCenterB = newCenterB
        self.__classCenterC = newCenterC

if __name__ == "__main__":
    a = np.array([[2, 0], [2, 4]])
    print(type(a[0]))
    print(np.shape(a)[0])
    print(calEuclideanDistance(a[0], a[1]))
    print(a[0] + a[1])
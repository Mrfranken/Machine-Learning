#encoding: utf-8

from logRegres import *
import numpy as np
from matplotlib import pyplot as plt


def plotDataSet():
    dataMat, labelMat = loadDataSet()          #加载数据集
    weights = gradAscent(dataMat, labelMat)

    dataArr = np.array(dataMat)                #转换成numpy的array数组
    n = np.shape(dataMat)[0]                   #数据个数
    xcord1 = []; ycord1 = []                   #正样本
    xcord2 = []; ycord2 = []                   #负样本
    for i in range(n):                         #根据数据集标签进行分类
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])    #1为正样本
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])    #0为负样本
    fig = plt.figure()
    ax = fig.add_subplot(111)                                           #添加subplot
    ax.scatter(xcord1, ycord1, s = 20, c = 'red', marker = 's',alpha=.5)#绘制正样本
    ax.scatter(xcord2, ycord2, s = 20, c = 'green',alpha=.5)            #绘制负样本

    x1 = np.arange(-4.0, 4.0, 0.1)
    x2 = (-weights[0] - weights[1]*x1) / weights[2]
    ax.plot(x1, x2.T)

    plt.title('DataSet')                                                #绘制title
    plt.xlabel('x')
    plt.ylabel('y')                                    #绘制label
    plt.show()


def stocGradAscent00(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


if __name__ == "__main__":
    plotDataSet()
    # dataMat, labelMat = loadDataSet()

    # stocGradAscent00(dataMat, labelMat)

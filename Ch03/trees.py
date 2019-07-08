#encoding: utf-8
'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''

from math import log
import operator
import copy
from treePlotter import retrieveTree, createPlot


def createDataSet():
    '''
    构建类型数据
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    '''
    计算熵
    '''
    # numEntries = len(dataSet)
    # labelCounts = {}
    # for featVec in dataSet:  # the the number of unique elements and their occurance
    #     currentLabel = featVec[-1]
    #     if currentLabel not in labelCounts.keys():
    #         labelCounts[currentLabel] = 0
    #     labelCounts[currentLabel] += 1
    # shannonEnt = 0.0
    # for key in labelCounts:
    #     prob = float(labelCounts[key]) / numEntries
    #     shannonEnt -= prob * log(prob, 2)  # log base 2
    # return shannonEnt
    labelCounts = dict()
    for item in dataSet:
        labelCounts[item[-1]] = labelCounts.get(item[-1], 0) + 1
    shannoEnt = 0
    for _, value in labelCounts.items():
        prob = float(value) / len(dataSet)
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt


def splitDataSet(dataSet, axis, value):
    # retDataSet = []
    # for featVec in dataSet:
    #     if featVec[axis] == value:
    #         reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
    #         reducedFeatVec.extend(featVec[axis + 1:])
    #         retDataSet.append(reducedFeatVec)
    # return retDataSet
    new_item = list()
    dataSet1 = copy.deepcopy(dataSet)
    for item in dataSet1:
        copy_item = item
        if value == copy_item[axis]:
            copy_item.pop(axis)
            new_item.append(copy_item)
    return new_item


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            sub_data_ent = calcShannonEnt(subDataSet)
            newEntropy += prob * sub_data_ent
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    '''
    构建决策树
    '''
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList) #直接投票，返回出现次数最多的标签
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''
    在验证算法的时候
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return:
    '''
    firstStr = inputTree.keys()[0] #当前树的根节点的特征名称
    secondDict = inputTree[firstStr] #根节点的所有子节点
    featIndex = featLabels.index(firstStr) #找到根节点特征对应的下标
    key = testVec[featIndex] #根据特征名称找到待测数据的特征值（特征值会对应一个分类）
    valueOfFeat = secondDict[key] #根据特征值得到对应的分类数据（有可能是一个分类也有可能是叶节点的分类数据）
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


# 序列化并保存在disk上
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


# 从disk上反序列化
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def splitDataSet1(dataset, axis, value):
    new_item = list()
    for item in dataset:
#         print(item)
        copy_item = item
        if value == copy_item[axis]:
            copy_item.pop(axis)
            print(copy_item)
            new_item.append(copy_item)
    return new_item


if __name__ == "__main__":
    """
    1 根据总的数据计算香农熵
    2 
    """
    dataSet, labels = createDataSet()
    calcShannonEnt(dataSet)
    outcome = splitDataSet(dataSet, 1, 1)
    print(outcome)
    print(chooseBestFeatureToSplit(dataSet))
    labels_copy = copy.deepcopy(labels)
    mytree = createTree(dataSet, labels_copy)
    print(mytree)

    ############ 测试算法 ############

    a = classify(mytree, labels, [1, 0])
    if a == 'no':
        print('successfully classify')
    a = classify(mytree, labels, [0, 0])
    if a == 'yes':
        print('successfully classify')

    ############ 测试pickle ############
    storeTree(mytree, r'D:\mytree.txt')
    outcome = grabTree(r'D:\mytree.txt')
    print(outcome)


    ############### generate tree from lenses.txt ###############
    fr = open('lenses.txt')
    lenses = [line.strip().split('\t') for line in fr.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    mytree = createTree(lenses, labels)
    print(mytree)
    createPlot(mytree)

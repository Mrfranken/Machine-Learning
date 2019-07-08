#encoding: utf-8

import math
import operator
from copy import deepcopy
from numpy import *


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


def cal_shannon_ent(data_set):
    label_count = dict()
    for item in data_set:
        label_count[item[-1]] = label_count.get(item[-1], 0) + 1

    shanno_ent = 0
    for _, value in label_count.items():
        prob = float(value) / len(data_set)
        shanno_ent -= prob * math.log(prob, 2)
    return shanno_ent


def split_data_set(data_set, axis, value):
    ret_data_set = []
    data_set_copy = deepcopy(data_set)
    for item in data_set_copy:
        if item[axis] == value:
            item.pop(axis)
            ret_data_set.append(item)

    return ret_data_set


def choose_best_feature_to_split(data_set):
    origin_shannon_ent = cal_shannon_ent(data_set) #计算原始数据的香农熵只要是标称型数据就行
    feature_num = len(data_set[0]) - 1 # 特征值的个数
    base_info_gain = 0.0
    base_feature = -1

    for axis in range(feature_num):
        unique_feature = set([item[axis] for item in data_set]) #根据特征值的位置开始遍历特征

        new_ent = 0
        for value in unique_feature:
            data_set_copy = deepcopy(data_set)
            sub_data_set = split_data_set(data_set_copy, axis, value) #根据特征进行数据的划分
            prob = len(sub_data_set) / float(len(data_set))

            child_shannon_ent = cal_shannon_ent(sub_data_set)
            new_ent += prob * child_shannon_ent

        info_gain = origin_shannon_ent - new_ent
        if info_gain > base_info_gain:
            base_info_gain = info_gain
            base_feature = axis

    return base_feature


def majority_class(class_list):
    class_count = {}
    for i in class_list:
        class_count[i] = class_count.get(i, 0) + 1
    a = sorted(class_count, key=operator.itemgetter(1), reverse=True)
    return a[0][0]


def create_tree(data_set, labels):
    class_list = [item[-1] for item in data_set]

    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_class(class_list)

    axis = choose_best_feature_to_split(data_set)
    best_feature_label = labels[axis]
    del labels[axis]
    tree = {best_feature_label: {}}
    feature_value = set([item[axis] for item in data_set])

    for value in feature_value:
        sub_labels = labels[:]
        tree[best_feature_label][value] = create_tree(split_data_set(data_set, axis, value), sub_labels)
    return tree


def classify(input_tree, feat_labels, test_vec):
    best_label = input_tree.keys()[0]
    seconde_dict = input_tree.get(best_label)
    best_label_index = feat_labels.index(best_label)
    value_of_feat = test_vec[best_label_index]
    for key in seconde_dict.keys():
        if key == value_of_feat:
            if type(seconde_dict[key]) is dict:
                class_label = classify(seconde_dict[key], feat_labels, test_vec)
            else:
                return seconde_dict[key]
    return class_label
    # if isinstance(value_of_feat, dict):
    #     classify(seconde_dict, feat_labels, test_vec)
    # else:
    #     class_label = value_of_feat
    # return class_label


def store_tree(input_tree, file_name):
    import pickle
    f = open(file_name, 'w')
    pickle.dumps(input_tree, f)
    f.close()


def get_tree(file_name):
    import pickle
    f = open(file_name)
    return pickle.dumps(f)


if __name__ == "__main__":
    data_set, labels = createDataSet()
    shanno_ent = cal_shannon_ent(data_set)
    print(shanno_ent)
    ret_data_set = split_data_set(data_set, 0, 0)
    print(ret_data_set)
    base_feature = choose_best_feature_to_split(data_set)
    print(base_feature)
    copy_labels = labels[:]
    my_tree = create_tree(data_set, copy_labels)
    print(my_tree)

    #根据决策树来分类数据数据一个类别
    class_label = classify(my_tree, labels, [1, 1])
    print(class_label)

    #read data from lens.txt
    f = open('lenses.txt')
    lenses = [line.strip().split('\t') for line in f.readlines()]
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_tree = create_tree(lenses, labels)
    print(lenses_tree)






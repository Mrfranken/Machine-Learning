# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import operator
import os

file_path = 'datingTestSet2.txt'


def file_to_matrix(file_path):
    # 这种写法太慢了，应该使用pandas去处理
    # f = open(file_path)
    # file_content = f.readlines()
    # file_row_length = len(file_content)
    # dating_mat_matrix = zeros((file_row_length, 3))
    # category_array = list()
    # index = 0
    #
    # for line in file_content:
    #     line_content = line.split('\t')
    #     data = line_content[:3]
    #     third_element = line_content[-1].split('\n')[0]
    #
    #     if third_element == 'largeDoses':
    #         category = 3
    #     elif third_element == 'smallDoses':
    #         category = 2
    #     else:
    #         category = 1
    #
    #     dating_mat_matrix[index, :] = data
    #     category_array.append(category)
    #     index += 1

    df = pd.read_table(file_path, sep='\t', names=['a', 'b', 'c', 'd'])
    dating_mat_matrix = df.ix[:, [0, 1, 2]]
    category_array = df.ix[:, 3]
    # category_array = list()
    # for third_element in df.ix[:, 3]:
    #     if third_element == 'largeDoses':
    #         category = 3
    #     elif third_element == 'smallDoses':
    #         category = 2
    #     else:
    #         category = 1
    #     category_array.append(category)
    return dating_mat_matrix, category_array


def auto_norm(data_set):
    set_min = data_set.min(0)
    ranges = data_set.max(0) - set_min
    norm_data_set = (data_set - set_min) / ranges
    return norm_data_set, ranges, set_min


def classify(inx, data_set, labels, k):
    traing_data_length = data_set.shape[0]
    diff_mat = np.tile(inx, (traing_data_length, 1)) - data_set
    distance = np.sum(diff_mat ** 2, axis=1) ** 0.5
    sorte_distance = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorte_distance[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def testing_data():
    returnMat, classLabelVector = file_to_matrix(file_path)
    norm_data_set, ranges, set_min = auto_norm(returnMat)
    mat_length = norm_data_set.shape[0]
    ratio = 0.5

    norm_mat = np.array(norm_data_set)
    labels = np.array(classLabelVector)
    limited_range = int(mat_length * ratio)

    testing_data = norm_mat[limited_range: mat_length]
    testing_labels = labels[limited_range: mat_length]

    error_num = 0
    for i in range(limited_range):
        classified_result = classify(norm_mat[i], testing_data, testing_labels, 3)
        if classified_result != labels[i]:
            error_num += 1

    error_ratio = error_num / float(limited_range)
    print( "the total error rate is: {}".format(error_ratio))


def classify_person():
    resultList = ['not at all', 'in small doses', 'in large doses']
    ffMils = float(raw_input("frequent filer miles earned per year?"));
    percentTats = float(raw_input("percentage of time spent playing video games ?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    returnMat, classLabelVector = file_to_matrix(file_path)
    norm_data_set, ranges, set_min = auto_norm(returnMat)
    in_arr = np.array([ffMils, percentTats, iceCream])

    norm_data_set = np.array(norm_data_set)
    ranges = np.array(ranges)
    set_min = np.array(set_min)

    classified_result = classify((in_arr - set_min) / ranges,
                                 np.array(norm_data_set),
                                 np.array(classLabelVector),
                                 3)
    print('you may like this person: ', resultList[classified_result - 1])


def image_to_vector(file_name):
    line_array = np.zeros((1, 1024))
    f = open(file_name)
    for line in range(32):
        content = f.readline() #注意！！！不要使用readlines，因为会带有'\n'符号，如果使用readline则没有
        for j in range(32):
            line_array[0, 32 * line + j] = int(content[j])
    return line_array


# def hand_writing_classify():
#     training_digit_files = os.listdir('digits/trainingDigits')
#
#     number_of_file = len(training_digit_files)
#     training_mat = np.zeros((number_of_file, 1024))
#
#     class_labels = list()
#     for file_name in training_digit_files:
#         class_number = int(file_name.split('_')[0])
#         class_labels.append(class_number)
#
#         file_index = training_digit_files.index(file_name)
#         training_mat[file_index, :] = image_to_vector('digits/trainingDigits/{}'.format(file_name))
#     # norm_data_set, ranges, set_min = auto_norm(training_mat)
#
#     test_digit_files = os.listdir('digits/testDigits')
#     number_of_test_file = len(test_digit_files)
#     error = 0
#     for i in range(number_of_test_file):
#         file_name = test_digit_files[i]
#
#         test_class_labels = int(file_name.split('_')[0])
#         test_vector = image_to_vector('digits/testDigits/{}'.format(file_name))
#
#         classified_number = classify(test_vector,
#                                      training_mat,
#                                      np.array(class_labels),
#                                      3)
#         if classified_number != test_class_labels:
#             error += 1.0
#             print(error)
#     print('error ratio is: ', error / float(number_of_test_file))

def hand_writing_classify():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = image_to_vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = os.listdir('digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = image_to_vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        # print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
            print(errorCount)
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    # returnMat, classLabelVector = file_to_matrix(file_path)
    # norm_data_set, ranges, set_min = auto_norm(returnMat)

    testing_data() #算法实现

    classify_person()#测试算法（实现任务喜欢度分类）

    # hand_writing_classify() # 测试算法（实现手写体数字分类）
    print(1)
    print(1)

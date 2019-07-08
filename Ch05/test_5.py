# -*- coding: utf-8 -*-
"""
machine learning logistic regression demo
"""
import numpy as np
import random


class SampleFeatureConvetor(object):
    """parse sample file and convert to feature"""
    _SEX_VALUE_MAPPING = {'male': 0, 'female': 1}
    _VALUE_SEX_MAPPING = {0: 'male', 1: 'female'}
    _CHAR_LIST = [chr(each) for each in range(ord('a'), ord('z') + 1)]

    def __init__(self):
        self._trainning_names = []
        self._trainning_sex_list = []
        self._trainning_x = None
        self._trainning_y = None
        self._test_names = []

    def parse_trainning_file(self, file_name):
        """
        parse training file
        """
        with open(file_name) as train_file:
            self._parse_trainning_sample(train_file.readlines())
            self._generate_trainning_matrix()

    def parse_test_file(self, file_name):
        """parse test file"""
        with open(file_name) as test_file:
            self._test_names = [line.strip() for line in test_file]

    def _parse_trainning_sample(self, lines):
        for line in lines:
            if line.strip():
                name, sex = line.split(',')
                self._trainning_names.append(name.strip())
                self._trainning_sex_list.append(self._SEX_VALUE_MAPPING[sex.strip()])

    def _generate_trainning_matrix(self):
        trainning_x = self._generate_trainning_x_matrix()
        self._trainning_x = np.array(trainning_x).reshape(len(trainning_x), len(trainning_x[0]))
        row_num = len(self._trainning_sex_list)
        self._trainning_y = np.array(self._trainning_sex_list).reshape(row_num, 1)

    def _generate_trainning_x_matrix(self):
        return [self._generate_feature(name) for name in self._trainning_names]

    def _generate_feature(self, name):
        statstic = dict(zip(self._CHAR_LIST, [0] * len(self._CHAR_LIST)))
        for each_char in list(name.lower()):
            if each_char in statstic:
                statstic[each_char] += 1
        return list(statstic.values())

    def generate_test_x_vector(self, name):
        """test input vector"""
        name_feature = self._generate_feature(name)
        return np.array(name_feature).reshape(1, len(name_feature))

    @property
    def train_x_matrix(self):
        """trainning input matrix"""
        return self._trainning_x

    @property
    def train_y_matrix(self):
        """trainning output matrix"""
        return self._trainning_y

    @property
    def test_names(self):
        """test name list"""
        return self._test_names

    def map_value_to_sex(self, sex_value):
        """map sex value to sex name"""
        return self._VALUE_SEX_MAPPING[sex_value]


class LogisticRegression(object):
    """logistic regression machine learning"""

    def __init__(self, learning_rate=0.0000025, max_cycle=40000):
        self._learning_rate = learning_rate
        self._max_cycle = max_cycle
        self._theta = 0

    @staticmethod
    def _sigmoid(exponent):
        return 1.0 / (1 + np.exp(-exponent))

    @staticmethod
    def _costfunction(output_y, hyp):
        output_y, hyp = np.array(output_y), np.array(hyp)
        return np.sum(output_y * np.log(hyp)) + np.sum((1 - output_y) * np.log(1 - hyp))

    def train(self, input_x, output_y):
        """train modle with sample"""
        row_num, col_num = input_x.shape
        # input_x = np.c_[np.ones(row_num), input_x]
        input_x, output_y = np.array(input_x), np.array(output_y)
        theta = np.ones(col_num)

        cost_list = []
        m = input_x.shape[0]
        for j in range(self._max_cycle):
            data_index = range(m)
            for i in range(m):
                alpha = 0.5/(2+j+i) + self._learning_rate
                rand_index = int(random.uniform(0, len(data_index)))
                h = self._sigmoid(np.dot(input_x[rand_index], theta))
                error = output_y[rand_index][0] - h
                theta = theta + alpha * error * input_x[rand_index]
                del data_index[rand_index]
        # hyp = self._sigmoid(input_x * theta)
        # theta = theta + self._learning_rate * (input_x.T) * (output_y - hyp)
        # cost = self._costfunction(output_y, hyp)
        # cost_list.append(cost)

        self._theta = theta

    def predict(self, input_vec):
        """predict output with test input"""
        prob = self._sigmoid(np.dot(input_vec, self._theta)[0])
        return int(np.where(prob >= 0.5, 1, 0))


if __name__ == "__main__":
    import time
    start = time.time()
    sample = SampleFeatureConvetor()
    sample.parse_trainning_file(r'D:\userdata\sijie\My Documents\MyJabberFiles\haofeng.zhu@nokia-sbell.com\5\c5.csv')
    sample.parse_test_file(r'D:\userdata\sijie\My Documents\MyJabberFiles\haofeng.zhu@nokia-sbell.com\5\intput.txt')

    logistic_regression = LogisticRegression(learning_rate=0.0000008, max_cycle=8000)
    logistic_regression.train(sample.train_x_matrix, sample.train_y_matrix)

    result = {}
    for test_name in sample.test_names:
        sex_value = logistic_regression.predict(sample.generate_test_x_vector(test_name))
        # print('{},{}'.format(test_name, sample.map_value_to_sex(sex_value)))
        result[test_name] = sample.map_value_to_sex(sex_value)

    print('cost:', time.time() - start)

    success_num = 0
    with open(r'D:\userdata\sijie\My Documents\MyJabberFiles\haofeng.zhu@nokia-sbell.com\5\intput_ref.txt') as ref_file:
        for each_line in ref_file:
            name, sex = [elem.strip() for elem in each_line.split(',')]
            if name in result:
                if result[name] == sex:
                    success_num += 1
    print('correction rate {}'.format(float(success_num)/float(len(result))))

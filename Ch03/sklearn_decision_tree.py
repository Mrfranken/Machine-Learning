# -*- coding: utf-8 -*-

from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.externals.six import StringIO
from sklearn.preprocessing import LabelEncoder
import pydotplus


if __name__ == "__main__":
    with open('lenses.txt') as f:
        lenses = [value.strip().split('\t') for value in f.readlines()]

    lenses_target = list()
    for i in lenses:
        lenses_target.append(i[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']            #特征标签
    lenses_list = []                                                        #保存lenses数据的临时列表
    lenses_dict = {}
    for each_label in lensesLabels:
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []

    lenses_pd = pd.DataFrame(lenses_dict)
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    clf = tree.DecisionTreeClassifier(max_depth=4, criterion='entropy')
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                        feature_names=lenses_pd.keys(),
                        filled=True,
                        rounded=True,
                        special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree_entropy1.pdf")





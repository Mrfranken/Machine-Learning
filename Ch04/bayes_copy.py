# encoding: utf-8
# from numpy import *
#
#
# def loadDataSet():
#     postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
#                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
#                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
#                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
#                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
#                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
#     classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
#     return postingList, classVec
#
#
# def create_vocab_list(data_set):
#     vocab_set = set([])
#     for item in data_set:
#         vocab_set = vocab_set | set(item)
#     return list(vocab_set)
#
#
# def set_words_to_vec(input_set, vocab_list):
#     return_vec = [0] * len(vocab_list)
#     for word in input_set:
#         if word in vocab_list:
#             return_vec[vocab_list.index(word)] = 1
#     return return_vec
#
#
# def train_algorithm(train_mat, list_classed):
#     spam_email_prob = sum(list_classed) / float(len(list_classed))
#
#     spam_vect = ones(len(train_mat[0]))
#     number_words_of_spam = 0
#     number_words_of_normal = 0
#     normal_vect = ones(len(train_mat[0]))
#
#     for i in range(len(list_classed)):
#         if list_classed[i] == 1:
#             spam_vect += train_mat[i]
#             number_words_of_spam += len(train_mat[i])
#         else:
#             normal_vect += train_mat[i]
#             number_words_of_normal += len(train_mat[i])
#
#     prob_of_spam = log(spam_vect / number_words_of_spam)
#     prob_of_normal = log(normal_vect / number_words_of_normal)
#
#     return spam_email_prob, prob_of_spam, prob_of_normal
#
# def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
#     p1 = sum(vec2Classify * p1Vec) + log(pClass1)  # element-wise mult
#     p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
#     if p1 > p0:
#         return 1
#     else:
#         return 0
#
# def testingNB():
#     postingList, classVec = loadDataSet()
#     vocab_list = create_vocab_list(postingList)
#     train_mat = []
#     for post_doc in postingList:
#         train_mat.append(set_words_to_vec(post_doc, vocab_list))
#     spam_email_prob, prob_of_spam, prob_of_normal = \
#         train_algorithm(train_mat, classVec)
#
#     testEntry = ['love', 'my', 'dalmation']
#     this_doc = array(set_words_to_vec(testEntry, vocab_list))
#     test_type = classifyNB(this_doc, prob_of_normal, prob_of_spam, spam_email_prob)
#     print(test_type)
#
#     testEntry = ['stupid', 'garbage']
#     this_doc = array(set_words_to_vec(testEntry, vocab_list))
#     test_type = classifyNB(this_doc, prob_of_normal, prob_of_spam, spam_email_prob)
#     print(test_type)
#
# if __name__ == "__main__":
#     postingList, classVec = loadDataSet()
#     print(postingList, classVec)
#
#     vocab_list = create_vocab_list(postingList)
#     print(vocab_list)
#
#     train_mat = []
#     for post_doc in postingList:
#         train_mat.append(set_words_to_vec(post_doc, vocab_list))
#
#     spam_email_prob, prob_of_spam, prob_of_normal = \
#         train_algorithm(train_mat, classVec)
#
#     testingNB()
#     print(1)
#
#
from numpy import *
import re


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def create_vocab_list(dataset):
    vocab_set = set([])
    for doc in dataset:
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bag_of_words_to_vec(vocab_list, input_set):
    return_vec = list(zeros(len(vocab_list)))
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_nb(train_matrix, train_category):
    prob_of_spam = sum(train_category) / float(len(train_category))
    num_of_train_data = len(train_matrix)
    num_of_words = len(train_matrix[0])
    spam_vect = ones(num_of_words)
    normal_vect = ones(num_of_words)

    total_spam_words, total_normal_words = 0, 0

    for i in range(num_of_train_data):
        if train_category[i] == 1:
            spam_vect += train_matrix[i]
            total_spam_words += sum(train_matrix[i])
        else:
            normal_vect += train_matrix[i]
            total_normal_words += sum(train_matrix[i])

    prob_of_spam_vect = log(spam_vect / float(total_spam_words))
    prob_of_normal_vect = log(normal_vect / float(total_normal_words))

    # prob_of_spam_vect = spam_vect / float(total_spam_words)
    # prob_of_normal_vect = normal_vect / float(total_normal_words)
    #
    return prob_of_spam_vect, prob_of_normal_vect, prob_of_spam

def classify(prob_of_spam_vect, prob_of_normal_vect, prob_of_spam, testing_vect):
    # posv = 1
    # ponv = 1
    # for item in testing_vect:
    #     if item:
    #         index = list(testing_vect).index(item)
    #         posv *= prob_of_spam_vect[index]
    #         ponv *= prob_of_normal_vect[index]
    # spam = posv * prob_of_spam
    # normal = ponv * (1 - prob_of_spam)
    #
    spam = sum(prob_of_spam_vect * testing_vect) + log(prob_of_spam)
    normal = sum(prob_of_normal_vect * testing_vect) + log(1 - prob_of_spam)
    if spam > normal:
        return 1
    else:
        return 0

def nb_testing():
    list_of_posts, list_of_classes = loadDataSet()
    vocabs_list = create_vocab_list(list_of_posts)
    train_mat = []
    for line in list_of_posts:
        train_mat.append(bag_of_words_to_vec(vocabs_list, line))
    prob_of_spam_vect, prob_of_normal_vect, prob_of_spam = train_nb(
        array(train_mat),
        array(list_of_classes)
    )

    test_vect = ['love', 'my', 'dalmation']
    testing_vect = array(setOfWords2Vec(vocabs_list, test_vect))
    result = classify(prob_of_spam_vect, prob_of_normal_vect, prob_of_spam, testing_vect)
    print(result)

    testEntry = ['stupid', 'garbage']
    testing_vect = array(setOfWords2Vec(vocabs_list, testEntry))
    result = classify(prob_of_spam_vect, prob_of_normal_vect, prob_of_spam, testing_vect)
    print(result)

def parse_text(file_path):
    contents = open(file_path).read()
    return [word.lower() for word in re.split(r'\W*', contents) if len(word) > 2]

def spam_email_test():
    wordList = []
    classList = []
    for i in range(1, 26):
        wordList.append(parse_text('email/spam/{}.txt'.format(i)))
        classList.append(1)
        wordList.append(parse_text('email/ham/{}.txt'.format(i)))
        classList.append(0)

    all_words_list = create_vocab_list(wordList)

    trainingSet = range(50)
    testSet = [] #训练集
    index_list = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        index_list.append(randIndex)
        del trainingSet[randIndex]

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bag_of_words_to_vec(all_words_list, wordList[docIndex]))
        trainClasses.append(classList[docIndex])

    prob_of_spam_vect, prob_of_normal_vect, prob_of_spam = train_nb(array(trainMat), array(trainClasses))
    error_count = 0
    for docIndex in trainingSet:
        test_vect = bag_of_words_to_vec(all_words_list, wordList[docIndex])
        email_type = classify(prob_of_spam_vect, prob_of_normal_vect, prob_of_spam, test_vect)
        if email_type != classList[docIndex]:
            error_count += 1
    print(error_count)
    print(float(error_count) / 10)


if __name__ == '__main__':
    nb_testing()
    # spam_email_test()

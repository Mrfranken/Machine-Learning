import csv
import numpy as np

all_words_list = [0] * 27

letter_list = []

train_mat = []


def generate_vect(line):
    every_test_l = np.array(all_words_list)
    for letter in line:
        location = ord(letter.lower()) - 97
        if location < 0:
            location = 26
        every_test_l[location] += 1
    return every_test_l


def load_data_set():
    class_list = []
    for line in csv.reader(open(r'D:\training_dataset.txt')):
        every_test_l = generate_vect(line[0])
        train_mat.append(every_test_l)
        if line[1] == 'male':
            class_list.append(1)
        else:
            class_list.append(0)

    return train_mat, class_list


def train_data(train_mat, train_class):
    prob_of_male = sum(train_class) / float(len(train_class))
    num_of_train_data = len(train_mat)
    num_of_words = len(train_mat[0])
    male_vect, female_vect = np.zeros(num_of_words), np.zeros(num_of_words)
    total_male_letters, total_female_letters = 0, 0

    for i in range(num_of_train_data):
        if train_class[i] == 1:
            male_vect += train_mat[i]
            total_male_letters += sum(train_mat[i])
        else:
            female_vect += train_mat[i]
            total_female_letters += sum(train_mat[i])

    prob_of_male_vect = np.log(male_vect / float(total_male_letters))
    prob_of_female_vect = np.log(male_vect / float(total_female_letters))
    # prob_of_male_vect = male_vect / float(total_male_letters)
    # prob_of_female_vect = male_vect / float(total_female_letters)
    prob_of_male = float(total_male_letters) / (total_male_letters + total_female_letters)
    return prob_of_male, prob_of_male_vect, prob_of_female_vect


def test_data(prob_of_male, prob_of_male_vect, prob_of_female_vect):
    file_path = r'D:\userdata\sijie\My Documents\MyJabberFiles\haofeng.zhu@nokia-sbell.com\5\intput_ref.txt'
    with open(file_path) as f:
        contents = f.readlines()

    test_data_sex_outcome = []
    test_data_sex = []

    male_prob = np.log(prob_of_male)
    female_prob = np.log(1 - prob_of_male)

    for line in contents:
        outcome = line.strip('\n').split(',')
        test_line_vect = generate_vect(outcome[0])
        male = np.dot(prob_of_male_vect, test_line_vect) + male_prob
        female = np.dot(prob_of_female_vect, test_line_vect) + female_prob

        if outcome[1] == 'male':
            test_data_sex.append(1)
        else:
            test_data_sex.append(0)

        if male >= female:
            test_data_sex_outcome.append(1)
        else:
            test_data_sex_outcome.append(0)

    count = 0
    for i in range(len(test_data_sex)):
        if test_data_sex_outcome[i] != test_data_sex[i]:
            count += 1

    print('count is: ', count)
    print('error rate is: ', count / float(len(test_data_sex)))


if __name__ == '__main__':
    a, b = load_data_set()
    prob_of_male, prob_of_male_vect, prob_of_female_vect = train_data(np.array(a), np.array(b))
    test_data(prob_of_male, prob_of_male_vect, prob_of_female_vect)





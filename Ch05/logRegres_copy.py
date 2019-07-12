from numpy import *


def loadDataSet():
    data_mat = []
    data_label = []
    f = open('testSet.txt')
    for line in f.readlines():
        line = line.strip().split()
        data_mat.append([1.0, float(line[0]), float(line[1])])
        data_label.append(int(line[2]))
    return data_mat, data_label


def sigmod(inx):
    return 1 / (1 + exp(-inx))


def gradAscent(data_mat, data_labels):
    data_matrix = mat(data_mat)
    label_matrix = mat(data_labels)
    m, n = data_matrix.shape
    alpha = 0.001
    weights = ones((n, 1))
    max_cycles = 500
    for i in range(max_cycles):
        h = sigmod(data_matrix * weights)  # 1000*3 3*1
        error = label_matrix.T - h  # 1000*1
        weights = weights + alpha * data_matrix.T * error
    return weights


def stochastic_gradient_descent(data_mat, data_labels):
    # data_matrix = array(data_mat)
    data_matrix = mat(data_mat)
    label_matrix = data_labels
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        # h = sigmod(sum(data_matrix[i]*weights)) #
        h = sigmod(weights * data_matrix[i].reshape(3, -1))[0, 0]
        error = label_matrix[i] - h
        weights = weights + alpha * error * data_matrix[i]
    # return array(weights)
    return array(weights)[0]


def plotBestFit():
    data_mat, data_label = loadDataSet()
    weights = gradAscent(data_mat, data_label)
    sto_weights = stochastic_gradient_descent(data_mat, data_label)
    num = len(data_label)
    x0, x1, y0, y1 = [], [], [], []
    for i in range(num):
        if int(data_label[i]) == 0:
            x0.append(data_mat[i][1])
            y0.append(data_mat[i][2])
        else:
            x1.append(data_mat[i][1])
            y1.append(data_mat[i][2])

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0, y0, s=30, c='red', marker='s')
    ax.scatter(x1, y1, s=30, c='green')

    x = arange(-5.0, 5.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    y1 = (-sto_weights[0] - sto_weights[1] * x) / sto_weights[2]

    ax.plot(x, y.T)
    ax.plot(x, y1)
    plt.show()


if __name__ == "__main__":
    data_mat, data_label = loadDataSet()
    weights = gradAscent(data_mat, data_label)
    plotBestFit()
    print(1)

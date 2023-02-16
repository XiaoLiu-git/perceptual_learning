import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import os

from Network import *
from representation_test import GenImg, representation


def test(net, inputs, num_batch, labels, prt=False):
    with torch.no_grad():
        b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)
        acc = ACC(outputs, labels, num_batch)
        if prt:
            print("Accuracy:")
            print(np.around(acc,3) * 100)
    return acc


def ACC(outputs, labels, num_batch):
    # pdb.set_trace()
    num_test = len(outputs) // num_batch
    acc = np.zeros(num_test + 1)

    total = (torch.sign(outputs * labels) + 1) / 2
    for i in range(num_test):
        acc[i] = torch.sum(total[num_batch * i:num_batch * (i + 1)]) / num_batch
    acc[-1] = torch.sum(total) / len(outputs)
    return acc


def np_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def np_acc(outputs, labels):
    output = outputs.detach().numpy()
    total = (np.sign(output * labels) + 1) / 2
    acc = np.sum(total) / len(output)
    return acc


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train(show_epc, net_name, slow_learning=True):
    # input
    num_batch = 16
    Img_L = GenImg(orient='V', loc="L", diff=0)
    Img_R = GenImg(orient='V', loc="R", diff=0)
    Img_DT = GenImg(orient='H', loc="R", diff=0)
    Img_th = GenImg(orient='H', loc="L", diff=0)
    inputs = np.zeros([num_batch, 1, 40, 18])
    labels = np.zeros([num_batch])

    # network
    if net_name is "s_cc3":
        net = Net_sCC()
    elif net_name is "s_cc5":
        net = Net_sCC5()
    elif net_name is "s_fc3":
        net = Net_sFC()
    else:
        # pdb.set_trace()
        net = Net_sCC7()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': 1e-3}
    ], lr=1e-3)
    running_loss = 0.0
    # generate testing data, select 40 < L,R,TH < 60
    acc_list = np.zeros([200 // show_epc + 1, 5])
    loss_list = np.zeros([200 // show_epc, 1])
    num_test = 20

    t_inputs = np.zeros([num_test * 4, 1, 40, 18])
    t_labels = np.zeros([num_test * 4])
    for i in range(num_test // 2):
        t_labels[2 * i] = 1
        t_labels[2 * i + 1] = -1
        # pdb.set_trace()
        img_tg = Img_L.gen_test()
        t_inputs[2 * i, :, :, :] = representation(img_tg[0])
        t_inputs[2 * i + 1, :, :, :] = representation(img_tg[1])

        t_labels[1 * num_test + 2 * i] = 1
        t_labels[1 * num_test + 2 * i + 1] = -1
        img_tg = Img_DT.gen_test()
        t_inputs[1 * num_test + 2 * i, :, :, :] = representation(img_tg[0])
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = representation(
            img_tg[1])

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1
        img_tg = Img_R.gen_test()
        t_inputs[2 * num_test + 2 * i, :, :, :] = representation(img_tg[0])
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = representation(
            img_tg[1])

        t_labels[3 * num_test + 2 * i] = 1
        t_labels[3 * num_test + 2 * i + 1] = -1
        img_tg = Img_th.gen_test()
        t_inputs[3 * num_test + 2 * i, :, :, :] = representation(img_tg[0])
        t_inputs[3 * num_test + 2 * i + 1, :, :, :] = representation(
            img_tg[1])

    _epc = 1
    size_input = t_inputs.shape
    # pdb.set_trace()
    t_inputs = np.reshape(normalize(np.reshape(t_inputs, [size_input[0], -1])),
                          size_input)
    acc_list[0, 1:] = test(net, t_inputs, num_test, t_labels)[:-1]
    acc_list[0, 0] = acc_list[0, 1]
    best_acc = np.zeros(np.size(acc_list[0, :]))

    # Training

    for epoch in range(200):  # loop over the dataset multiple times

        for i in range(num_batch):
            labels[i], img_tg = Img_L.gen_train()
            inputs[i, :, :, :] = representation(img_tg)

        size_input = inputs.shape
        inputs = np.reshape(normalize(np.reshape(inputs, [num_batch, -1])),
                            size_input)

        optimizer.zero_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, torch.FloatTensor((labels + 1) // 2))
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch % show_epc == show_epc - 1:  # print every show epoch
            print_test = False
            acc = np_acc(outputs, labels)
            acc_list[_epc, 0] = acc
            loss_list[_epc - 1, 0] = running_loss / show_epc
            if epoch % (show_epc * 10) == show_epc * 10 - 1:
                print('[%d] loss: %.6f' %
                      (epoch + 1, running_loss / show_epc))
                print('train acc: %.2f %%' % (acc * 100))
                print_test = True
            PATH = './net.pth'
            torch.save(net.state_dict(), PATH)
            running_loss = 0.0
            acc_test =test(net, t_inputs, num_test, t_labels,print_test)
            acc_list[_epc, 1:] = acc_test[:-1]
            if acc_test[-1]>best_acc[-1]:
                best_net = net
                best_acc = acc_test
            _epc += 1

        if epoch == 199 and slow_learning:
            # pdb.set_trace()
            X_train = np.reshape(t_inputs[:100, :, :, ], (100, -1))
            X_tpe = np.reshape(t_inputs[:200, :, :, ], (200, -1))
            X_test = np.reshape(t_inputs, (400, -1))
            # pdb.set_trace()

            plt.figure()
            test_loc_size = [1, 3, 5, 7]
            for loc_size in range(len(test_loc_size)):
                print("size of local connection is " + str(
                    test_loc_size[loc_size]))
                Y_train = normalize(
                    loc_feedforward(X_train, X_test, test_loc_size[loc_size]))
                Y_tpe = normalize(
                    loc_feedforward(X_tpe, X_test, test_loc_size[loc_size]))
                plt.subplot(4, 2, loc_size * 2 + 1)
                plt.imshow(Y_train)
                plt.subplot(4, 2, loc_size * 2 + 2)
                plt.imshow(Y_tpe)
                Y_train = np.reshape(Y_train, (400, 1, 40, 18))
                Y_tpe = np.reshape(Y_tpe, (400, 1, 40, 18))
                acc_hebb_train = test(net, Y_train, num_test, t_labels,
                                      print_test)[:-1]
                acc_hebb_tpe = test(net, Y_tpe, num_test, t_labels,
                                    print_test)[:-1]
            plt.tight_layout()
            plt.show()

            print("loc_Hebb")
            # pdb.set_trace()

            # plt.figure()
            # plt.subplot(231)
            # plt.imshow(X_test[0, :].reshape([40, 18]))
            # plt.subplot(234)
            # plt.imshow(X_test[101, :].reshape([40, 18]))
            # plt.subplot(232)
            # plt.imshow(Y3_train[0, 0, :])
            # plt.subplot(235)
            # plt.imshow(Y3_train[101, 0, :])
            # plt.subplot(233)
            # plt.imshow(Y3_tpe[0, 0, :])
            # plt.subplot(236)
            # plt.imshow(Y3_tpe[101, 0, :])
            # plt.show()
            #
            # plt.figure()
            # plt.subplot(231)
            # plt.imshow(X_test[0, :].reshape([40, 18]))
            # plt.subplot(234)
            # plt.imshow(X_test[101, :].reshape([40, 18]))
            # plt.subplot(232)
            # plt.imshow(Y5_train[0, 0, :])
            # plt.subplot(235)
            # plt.imshow(Y5_train[101, 0, :])
            # plt.subplot(233)
            # plt.imshow(Y5_tpe[0, 0, :])
            # plt.subplot(236)
            # plt.imshow(Y5_tpe[101, 0, :])
            # plt.show()

    print("Best Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(
        best_acc[0] * 100,
        best_acc[1] * 100,
        best_acc[2] * 100,
        best_acc[3] * 100,
        best_acc[4] * 100))
    return acc_list, loss_list, best_net


def pca(X, Y):
    # calculate the mean vector
    mean_vector = X.mean(axis=0)

    # calculate the covariance matrix。协方差矩阵是对称矩阵，行数和列数为特征的个数。
    cov_mat = np.cov((X - mean_vector).T)

    # 计算协方差矩阵的特征值
    # calculate the eigenvectors and eigenvalues of our covariance matrix of the iris dataset
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)

    # 用保留的特征向量来变换原数据，生成新的数据矩阵
    # store the top two eigenvectors in a variable。假如这里选定了前两个特征向量。
    top_5_eigenvectors = eig_vec_cov[:, :5].T
    # show the transpose so that each row is a principal component, we have two rows == two

    # we will multiply the matrices of our data and our eigen vectors together
    Z = np.dot(Y, top_5_eigenvectors.T)
    return np.dot(Z, top_5_eigenvectors)


def hebb(x, y):
    mean = np.mean(x)
    sigma = np.std(x)
    norm_x = (x - mean) / sigma
    max_x = np.max(norm_x)
    norm_x = norm_x / max_x
    W = np.dot(norm_x.T, norm_x) / x.shape[0]
    norm_y = np.dot(y, W)
    return norm_y * max_x * sigma + mean


def loc_hebb(x, y, n=3):
    # pdb.set_trace()
    w_size = x.shape[0]
    mean = np.mean(x)
    sigma = np.std(x)
    norm_x = (x - mean) / sigma
    max_x = np.max(norm_x)
    norm_x = norm_x / max_x
    W = np.dot(norm_x.T, norm_x) / w_size
    norm_y = np.zeros(y.shape)
    nb_list = neighbor_list((40, 18), n)
    for i in range(y.shape[1]):
        norm_y[:, i] = np.dot(y[:, nb_list[i]], W[nb_list[i], i])
    return norm_y * max_x * sigma + mean


def loc_feedforward(x, y, n=3):
    # pdb.set_trace()
    if n == 1:
        # pdb.set_trace()
        w = np.var(x, axis=0)
        norm_y = y * w
    else:
        nb_list = neighbor_list((40, 18), n)
        w = np.var(x, axis=0)
        norm_y = np.zeros(y.shape)
        for i in range(y.shape[1]):
            norm_y[:, i] = np.dot(y[:, nb_list[i]], w[nb_list[i]])
    # pdb.set_trace()
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(y)
    # plt.subplot(2,1,2)
    # plt.imshow(norm_y)
    # plt.show()
    return norm_y


def neighbors(arr, x, y, n=3):
    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
    arr = np.roll(np.roll(arr, shift=-x + 1, axis=0), shift=-y + 1, axis=1)
    return arr[:n, :n]


def neighbor_list(size_2d, n):
    # pdb.set_trace()
    nb_list = []
    size0, size1 = size_2d
    Id = np.reshape(range(size0 * size1), [size0, size1])
    for i in range(size0):
        for j in range(size1):
            _list = np.reshape(neighbors(Id, i, j, n).astype(int), [-1])
            nb_list.append(_list)
    return nb_list


def normalize(x):
    mean = np.expand_dims(np.mean(x, axis=1), 1)
    sigma = np.expand_dims(np.std(x, axis=1), 1)
    sigma[sigma == 0] = 1
    nor_x = (x - mean) / sigma
    return nor_x


def main():
    AccALL = np.zeros([50, 101, 5])
    LossALL = np.zeros([50, 100, 1])
    net_list = ['s_cc3']
    plt.rcParams['font.sans-serif'] = [
        'Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.rcParams['svg.fonttype'] = 'none'
    for net_name in net_list:
        AccALL[0, :, :], LossALL[0, :, :], DL_net = train(show_epc=2,
                                                          net_name=net_name)
        print('====================================\n')
        print('Finish Decision Layer')
        Acc = np.squeeze(AccALL[0, :, :])
        Loss = np.squeeze(LossALL[0, :, :])

        plt.style.available
        plt.style.use('seaborn')
        # pdb.set_trace()
        plt.figure(figsize=(20, 12), dpi=80)

        plt.subplot(221)
        plt.plot(np.arange(1, 200, 2), Loss, linewidth=4)

        plt.title("Train with " + net_name, fontsize=24)
        plt.ylabel("Loss", fontsize=18)
        plt.xlabel("No. epoch", fontsize=18)

        plt.subplot(222)
        plt.plot(np.arange(0, 202, 2), Acc * 100, linewidth=4,
                 label=["train", "ori1_loc1", "ori2_loc2", "ori1_loc2",
                        "ori2_loc1"])

        plt.title("Train with " + net_name, fontsize=24)
        plt.ylabel("Accuracy %", fontsize=18)
        plt.xlabel("No. epoch", fontsize=18)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('./1122/' + net_name + '.jpg')


if __name__ == "__main__":
    main()

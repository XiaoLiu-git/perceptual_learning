import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from Network import *
from representation_test import GenImg, representation


def test(net, inputs, num_batch, labels, prt=False):
    with torch.no_grad():
        b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)
        acc = ACC(outputs, labels, num_batch)
        if prt:
            print("Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(acc[0] * 100,
                                                                 acc[1] * 100,
                                                                 acc[2] * 100,
                                                                 acc[3] * 100))
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


def train(show_epc, net_name):
    # input
    num_batch = 16
    Img_L = GenImg(orient='V', loc="L", diff=1)
    Img_R = GenImg(orient='V', loc="R", diff=1)
    Img_th = GenImg(orient='H', loc="L", diff=1)
    inputs = np.zeros([num_batch, 1, 40, 18])
    labels = np.zeros([num_batch])

    # network
    if net_name is "s_cc3":
        net = Net_sCC()
    elif net_name is "s_cc5":
        net = Net_sCC5()
    else:
        # pdb.set_trace()
        net = Net_sCC7()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': 1e-4}
    ], lr=1e-4)
    running_loss = 0.0
    # generate testing data, select 40 < L,R,TH < 60
    acc_list = np.zeros([200 // show_epc + 1, 4])
    loss_list = np.zeros([200 // show_epc, 1])
    num_test = 100

    t_inputs = np.zeros([num_test * 3, 1, 40, 18])
    t_labels = np.zeros([num_test * 3])
    for i in range(num_test // 2):
        t_labels[2 * i] = 1
        t_labels[2 * i + 1] = -1
        # pdb.set_trace()
        img_tg = Img_L.gen_test()
        t_inputs[2 * i, :, :, :] = representation(img_tg[0])
        t_inputs[2 * i + 1, :, :, :] = representation(img_tg[1])

        t_labels[1 * num_test + 2 * i] = 1
        t_labels[1 * num_test + 2 * i + 1] = -1
        img_tg = Img_R.gen_test()
        t_inputs[1 * num_test + 2 * i, :, :, :] = representation(img_tg[0])
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = representation(
            img_tg[1])

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1
        img_tg = Img_th.gen_test()
        t_inputs[2 * num_test + 2 * i, :, :, :] = representation(img_tg[0])
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = representation(
            img_tg[1])

    _epc = 1
    acc_list[0, 1:] = test(net, t_inputs, num_test, t_labels)[:-1]
    acc_list[0, 0] = acc_list[0, 1]

    # Training

    for epoch in range(200):  # loop over the dataset multiple times

        for i in range(num_batch):
            labels[i], img_tg = Img_L.gen_train()
            inputs[i, :, :, :] = representation(img_tg)

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
            acc_list[_epc, 1:] = test(net, t_inputs, num_test, t_labels,
                                      print_test)[:-1]
            _epc += 1
    return acc_list, loss_list, net

def normalize(x):
    sigma = np.std(x,axis=0)
    return x/sigma

def hebbian_learning(rules, net, show_epc=2):
    Img_L = GenImg(orient='V', loc="L", diff=1)
    Img_R = GenImg(orient='V', loc="R", diff=1)
    Img_th = GenImg(orient='H', loc="L", diff=1)

    # Initialization for Oja # w和q两种更新策略分别是recurrent & feedforward
    W = normalize(np.random.rand(40*18, 40*18)) # weight x->x
    Q = normalize(np.random.rand(40*18, 40*18)) # weight x->y

    # # Initialization for BCM # w和q两种更新策略分别是recurrent & feedforward
    # W = normalize(np.random.rand(40*18, 40*18)) # weight x->x
    # Q = normalize(np.random.rand(40*18, 40*18)) # weight x->y
    # # Initialization for anti-Hebb #
    # W = normalize(np.random.rand(40*18, 40*18)) # weight x->x
    # Q = np.zeros([40*18, 40*18]) # weight x->y
    #
    lr = 0.01
    running_loss = 0.0
    inputs = np.zeros()
    criterion = torch.nn.BCEWithLogitsLoss()
    acc_list = np.zeros([200 // show_epc + 1, 4])
    loss_list = np.zeros([200 // show_epc, 1])
    num_test = 100
    t_inputs = np.zeros([num_test * 3, 1, 40, 18])
    t_labels = np.zeros([num_test * 3])
    for i in range(num_test // 2):
        t_labels[2 * i] = 1
        t_labels[2 * i + 1] = -1
        # pdb.set_trace()
        img_tg = Img_L.gen_test()
        reprs0 = representation(img_tg[0])
        reprs1 = representation(img_tg[1])
        t_inputs[2 * i, :, :, :] = reprs0
        t_inputs[2 * i + 1, :, :, :] = reprs1

        t_labels[1 * num_test + 2 * i] = 1
        t_labels[1 * num_test + 2 * i + 1] = -1

        t_inputs[1 * num_test + 2 * i, :, :, :] = np.roll(reprs0, -5, axis=0)
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = np.roll(reprs1, -5,
                                                              axis=0)

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1

        t_inputs[2 * num_test + 2 * i, :, :, :] = np.roll(reprs0, -4, axis=1)
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = np.roll(reprs1, -4,
                                                              axis=1)

    for epoch in range(200):
        label, img_tg = Img_L.gen_train()
        x = representation(img_tg).flatten()
        x_ = np.dot(W,np.dot(W,np.dot(W,np.dot(W,np.dot(W, x))))) #用5次做收敛
        y = np.dot(Q,x)
        # Oja update
        W = W + lr*(np.dot(x_.T,x)-W*x_**2)
        Q = Q + lr*(np.dot(y.T,x)-W*y**2)
        # BCM update
        W = W + lr * (np.dot((x_**2).T,x) - np.dot(x_.T,x))
        Q = Q + lr * (np.dot((y ** 2).T, x) - np.dot(y.T, x))

        # PCA ###TODO!!!!


        inputs[0, :, :] = np.reshape(x_, [40, 18])
        inputs[1, :, :] = np.reshape(y, [40, 18])
        labels = np.dot(np.ones([2, 1]), label)

        torch.no_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, torch.FloatTensor((labels + 1) // 2))
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
            acc_list[_epc, 1:] = test(net, t_inputs, num_test, t_labels,
                                      print_test)[:-1]
            _epc += 1



AccALL = np.zeros([50, 101, 4])
LossALL = np.zeros([50, 100, 1])
net_list = ['s_fc3']
for net_name in net_list:
    AccALL[0, :, :], LossALL[0, :, :], DL_net = train(show_epc=2, net_name=net_name)
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
             label=["train", "test", "loc2", "ori2"])

    plt.title("Train with " + net_name, fontsize=24)
    plt.ylabel("Accuracy %", fontsize=18)
    plt.xlabel("No. epoch", fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig('./1027/' + net_name + '.jpg')


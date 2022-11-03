import pdb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from Network import *
from Representation import GenImg, representation


def test(net, inputs, num_batch, labels, prt=False):
    with torch.no_grad():
        b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)
        acc = ACC(outputs, labels, num_batch)
        if prt:
            print(
                "Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}% ".format(
                    acc[0] * 100,
                    acc[1] * 100,
                    acc[2] * 100,
                    acc[3] * 100,
                    acc[4] * 100,
                    acc[5] * 100,
                    acc[6] * 100
                ))
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
    inputs = np.zeros([num_batch, 1, 40, 18])
    labels = np.zeros([num_batch])

    # network
    if net_name is "s_cc":
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
    acc_list = np.zeros([200 // show_epc + 1, 7])
    loss_list = np.zeros([200 // show_epc, 1])
    num_test = 100

    t_inputs = np.zeros([num_test * 6, 1, 40, 18])
    t_labels = np.zeros([num_test * 6])
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

        t_inputs[1 * num_test + 2 * i, :, :, :] = np.roll(reprs0, 1, axis=0)
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = np.roll(reprs1, 1, axis=0)

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1

        t_inputs[2 * num_test + 2 * i, :, :, :] = np.roll(reprs0, 5, axis=0)
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = np.roll(reprs1, 5, axis=0)

        t_labels[3 * num_test + 2 * i] = 1
        t_labels[3 * num_test + 2 * i + 1] = -1

        t_inputs[3 * num_test + 2 * i, :, :, :] = np.roll(reprs0, 10, axis=0)
        t_inputs[3 * num_test + 2 * i + 1, :, :, :] = np.roll(reprs1, 10, axis=0)

        t_labels[4 * num_test + 2 * i] = 1
        t_labels[4 * num_test + 2 * i + 1] = -1

        t_inputs[4 * num_test + 2 * i, :, :, :] = np.roll(reprs0, 15, axis=0)
        t_inputs[4 * num_test + 2 * i + 1, :, :, :] = np.roll(reprs1, 15,
                                                              axis=0)

        t_labels[5 * num_test + 2 * i] = 1
        t_labels[5 * num_test + 2 * i + 1] = -1

        t_inputs[5 * num_test + 2 * i, :, :, :] = np.roll(reprs0, 20, axis=0)
        t_inputs[5 * num_test + 2 * i + 1, :, :, :] = np.roll(reprs1, 20,
                                                              axis=0)

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
    return acc_list, loss_list


AccALL = np.zeros([50, 101, 7])
LossALL = np.zeros([50, 100, 1])
net_list = ['s_cc','s_cc5']
for net_name in net_list:
    for num_train in range(50):
        AccALL[num_train, :, :], LossALL[num_train, :, :] = train(show_epc=2,
                                                                  net_name=net_name)
    print('====================================\n')
    print('Finish ' + net_name)
    Acc = np.squeeze(np.mean(AccALL, axis=0))
    Loss = np.squeeze(np.mean(LossALL, axis=0))
    np.save('./0928/' + net_name + '_acc.npy', AccALL)
    np.save('./0928/' + net_name + '_loss.npy', LossALL)
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
             label=["train", "test", "loc_1", "loc_5", "loc_10", "loc_15", "loc_20"])
    plt.title("Train with " + net_name, fontsize=24)
    plt.ylabel("Accuracy %", fontsize=18)
    plt.xlabel("No. epoch", fontsize=18)
    plt.legend(loc='upper left')
    plt.savefig('./0928/' + net_name + '.jpg')

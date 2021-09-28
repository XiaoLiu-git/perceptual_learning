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
    if net_name is "CC":
        net = Net_cc()
    elif net_name is "CF":
        net = Net_cf()
    elif net_name is "FC":
        net = Net_fc()
    else:
        net = Net_ff()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': 1e-7}
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
    acc_list[0, 1:] = test(net, t_inputs, num_test, t_labels)[:3]
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
            print_test=False
            acc = np_acc(outputs, labels)
            acc_list[_epc, 0] = acc
            loss_list[_epc - 1, 0] = running_loss / show_epc
            if epoch % (show_epc * 10) == show_epc * 10 - 1:
                print('[%d] loss: %.6f' %
                      (epoch + 1, running_loss / show_epc))
                print('train acc: %.2f %%' % (acc * 100))
                print_test=True
            PATH = './net.pth'
            torch.save(net.state_dict(), PATH)
            running_loss = 0.0
            acc_list[_epc, 1:] = test(net, t_inputs, num_test, t_labels,print_test)[:3]
            _epc += 1
    return acc_list, loss_list


AccALL = np.zeros([20, 101, 4])
LossALL = np.zeros([20, 100, 1])
net_list = ['CC', 'CF', 'FC', 'FF']
for net_name in net_list:
    for num_train in range(20):
        AccALL[num_train, :, :], LossALL[num_train, :, :] = train(show_epc=2,
                                                                  net_name=net_name)
    print('====================================\n')
    print('Finish ' + net_name)
    Acc = np.squeeze(np.mean(AccALL, axis=0))
    Loss = np.squeeze(np.mean(LossALL, axis=0))
    np.save('./0926/'+net_name + '_acc.npy', AccALL)
    np.save('./0926/'+net_name + '_loss.npy', LossALL)
    plt.style.available
    plt.style.use('seaborn')
    plt.rcParams.update({'font.size': 40})
    # pdb.set_trace()
    plt.figure(figsize=(20, 12), dpi=80)

    plt.subplot(221)
    plt.plot(np.arange(1, 200, 2), Loss, linewidth=4)

    plt.title("Train with " + net_name, fontsize=24)
    plt.ylabel("Loss")
    plt.xlabel("No. epoch")

    plt.subplot(222)
    plt.plot(np.arange(0, 202, 2), Acc * 100, linewidth=4,
             label=["train", "test", "loc2", "ori2"])
    plt.title("Train with " + net_name, fontsize=24)
    plt.ylabel("Accuracy %")
    plt.xlabel("No. epoch")
    plt.legend(loc='upper left')
    plt.savefig('./0926/'+net_name +'.jpg')

import pdb
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

import offline_Hebb as ob
import feature_learning as fl


def GenTestData(_ori, num=10, loc="L", diff=0):
    num_ori = int(180 / _ori)
    dataset = np.zeros([num * 2 * num_ori, 1, 40, 18])
    label = np.zeros([num * 2 * num_ori])
    for i in range(num_ori):
        ori_i = i * _ori
        Img = ob.GenImg(orient=ori_i, loc=loc, diff=diff)
        for ii in range(num):
            label[i * num * 2 + ii] = 1
            label[i * num * 2 + ii + num] = -1
            img = Img.gen_test()
            dataset[i * num * 2 + ii, :, :, :] = ob.representation(img[0])
            dataset[i * num * 2 + ii + num, :, :, :] = ob.representation(
                img[1])
    return dataset, label


def ff_train(show_epc, net_name, slow_learning=True, bg_epoch=0):
    # input
    num_batch = 16
    Img_L = ob.GenImg(orient='V', loc="L", diff=0)
    Img_th = ob.GenImg(orient='H', loc="L", diff=0)
    inputs = np.zeros([num_batch, 1, 40, 18])
    labels = np.zeros([num_batch])

    # network
    if net_name is "s_cc3":
        net = ob.Net_sCC()
    elif net_name is "s_cc5":
        net = ob.Net_sCC5()
    elif net_name is "s_fc3":
        net = ob.Net_sFC()
    else:
        # pdb.set_trace()
        net = ob.Net_sCC7()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': 1e-3}
    ], lr=1e-3)
    running_loss = 0.0
    # generate testing data, select 40 < L,R,TH < 60
    acc_list = np.zeros([200 // show_epc + 1, 10])
    loss_list = np.zeros([200 // show_epc, 1])
    num_test = 20

    t_inputs = np.zeros([num_test * 9, 1, 40, 18])
    t_labels = np.zeros([num_test * 9])
    for i in range(num_test // 2):
        t_labels[2 * i] = 1
        t_labels[2 * i + 1] = -1
        # pdb.set_trace()
        img_tg = Img_L.gen_test()
        t_inputs[2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[2 * i + 1, :, :, :] = ob.representation(img_tg[1])

        t_labels[1 * num_test + 2 * i] = 1
        t_labels[1 * num_test + 2 * i + 1] = -1
        img_tg = Img_th.gen_test()
        t_inputs[1 * num_test + 2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = ob.representation(
            img_tg[1])

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1
        img_tg = Img_L.gen_test()
        t_inputs[2 * num_test + 2 * i, :, :, :] = np.roll(ob.representation(
            img_tg[0]),5, axis=0)
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = np.roll(ob.representation(
            img_tg[1]),5, axis=0)

        t_labels[3 * num_test + 2 * i] = 1
        t_labels[3 * num_test + 2 * i + 1] = -1
        img_tg = Img_L.gen_test()
        t_inputs[3 * num_test + 2 * i, :, :, :] = np.roll(ob.representation(
            img_tg[0]), 10, axis=0)
        t_inputs[3 * num_test + 2 * i + 1, :, :, :] = np.roll(ob.representation(
            img_tg[1]), 10, axis=0)

        t_labels[4 * num_test + 2 * i] = 1
        t_labels[4 * num_test + 2 * i + 1] = -1
        img_tg = Img_L.gen_test()
        t_inputs[4 * num_test + 2 * i, :, :, :] = np.roll(ob.representation(
            img_tg[0]), 15, axis=0)
        t_inputs[4 * num_test + 2 * i + 1, :, :, :] = np.roll(ob.representation(
            img_tg[1]), 15, axis=0)

        t_labels[5 * num_test + 2 * i] = 1
        t_labels[5 * num_test + 2 * i + 1] = -1
        img_tg = Img_L.gen_test()
        t_inputs[5 * num_test + 2 * i, :, :, :] = np.roll(ob.representation(
            img_tg[0]), 20, axis=0)
        t_inputs[5 * num_test + 2 * i + 1, :, :, :] = np.roll(ob.representation(
            img_tg[1]), 20, axis=0)

        t_labels[6 * num_test + 2 * i] = 1
        t_labels[6 * num_test + 2 * i + 1] = -1
        img_tg = Img_L.gen_test()
        t_inputs[6 * num_test + 2 * i, :, :, :] = np.roll(ob.representation(
            img_tg[0]), 25, axis=0)
        t_inputs[6 * num_test + 2 * i + 1, :, :, :] = np.roll(ob.representation(
            img_tg[1]), 25, axis=0)

        t_labels[7 * num_test + 2 * i] = 1
        t_labels[7 * num_test + 2 * i + 1] = -1
        img_tg = Img_L.gen_test()
        t_inputs[7 * num_test + 2 * i, :, :, :] = np.roll(ob.representation(
            img_tg[0]), 30, axis=0)
        t_inputs[7 * num_test + 2 * i + 1, :, :, :] = np.roll(ob.representation(
            img_tg[1]), 30, axis=0)

        t_labels[8 * num_test + 2 * i] = 1
        t_labels[8 * num_test + 2 * i + 1] = -1
        img_tg = Img_L.gen_test()
        t_inputs[8 * num_test + 2 * i, :, :, :] = np.roll(ob.representation(
            img_tg[0]), 35, axis=0)
        t_inputs[8 * num_test + 2 * i + 1, :, :, :] = np.roll(ob.representation(
            img_tg[1]), 35, axis=0)

    _epc = 1
    t_inputs = fl.norma_rep(t_inputs)

    acc_list[0, 1:] = ob.test(net, t_inputs, num_test, t_labels)[:-1]
    acc_list[0, 0] = acc_list[0, 1]
    best_acc = np.zeros(np.size(acc_list[0, :]))

    # Training
    weight = np.ones([40, 1])
    for epoch in range(200):  # loop over the dataset multiple times

        for i in range(num_batch):
            labels[i], img_tg = Img_L.gen_train()
            inputs[i, :, :, :] = ob.representation(img_tg)

        inputs = fl.norma_rep(inputs)

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
            acc = ob.np_acc(outputs, labels)
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
            t_input_ff = fl.feedforward(t_inputs, weight)
            t_input_ff = fl.norma_rep(t_input_ff)
            acc_test = ob.test(net, t_input_ff, num_test, t_labels, print_test)
            acc_list[_epc, 1:] = acc_test[:-1]
            if acc_test[-1] > best_acc[-1]:
                best_net = net
                best_acc = acc_test
            _epc += 1

    print("Best Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(
        best_acc[0] * 100,
        best_acc[1] * 100,
        best_acc[2] * 100,
        best_acc[3] * 100,
        best_acc[4] * 100))
    return acc_list, loss_list, net, weight


# plt.style.available
# plt.style.use('seaborn')
day = datetime.now().strftime('%Y-%m-%d')
foldername = './' + day
ob.mkdir(foldername)
num_section = 20
num_test = 10
ori = 22.5
loc = 5
AccALL = np.zeros([num_section, 101, 10])
LossALL = np.zeros([num_section, 100, 1])
Acc_test = np.zeros([num_section, int(40 / loc), int(180 / ori)])
print_test = False
net_list = ['s_cc3']
t_data, t_label = GenTestData(_ori=ori)
t_data = fl.norma_rep(t_data)

# pdb.set_trace()

for net_name in net_list:
    for begin_epoch in [1]:
        for s in range(num_section):
            ### Training
            AccALL[s, :, :], LossALL[s, :, :], net, weight = ff_train(
                show_epc=2,
                net_name=net_name,
                slow_learning=False,
                bg_epoch=begin_epoch)
            print(AccALL[s, :, :])

            ### Testing for different location

            for i in range(40 // loc):
                y = fl.feedforward(t_data, weight)
                y = fl.norma_rep(y)

                # plt.subplot(2, 4, i+1)
                # plt.imshow(y[1].squeeze())
                Acc_test[s, i, :] = ob.test(net, y, num_test * 2, t_label,
                                            print_test)[:-1]
                t_data = np.roll(t_data, -5, axis=2)
            print(Acc_test[s])

        name_head = foldername + '/task_module' + net_name
        np.save(name_head + net_name + 'AccALL.npy', AccALL)
        np.save(name_head + 'LossALL.npy', LossALL)
        np.save(name_head + 'Acc_test.npy', Acc_test)


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



def first_train(show_epc, net_name, bg_epoch=0):
    # input
    num_batch = 16
    Img_L = ob.GenImg(orient='V', loc="L", diff=0)
    Img_R = ob.GenImg(orient='V', loc="R", diff=0)
    Img_DT = ob.GenImg(orient='H', loc="R", diff=0)
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
    acc_list = np.zeros([bg_epoch//show_epc + 1, 5])
    loss_list = np.zeros([bg_epoch // show_epc, 1])
    num_test = 20

    t_inputs = np.zeros([num_test * 4, 1, 40, 18])
    t_labels = np.zeros([num_test * 4])
    for i in range(num_test // 2):
        t_labels[2 * i] = 1
        t_labels[2 * i + 1] = -1
        # pdb.set_trace()
        img_tg = Img_L.gen_test()
        t_inputs[2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[2 * i + 1, :, :, :] = ob.representation(img_tg[1])

        t_labels[1 * num_test + 2 * i] = 1
        t_labels[1 * num_test + 2 * i + 1] = -1
        img_tg = Img_DT.gen_test()
        t_inputs[1 * num_test + 2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = ob.representation(
            img_tg[1])

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1
        img_tg = Img_R.gen_test()
        t_inputs[2 * num_test + 2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = ob.representation(
            img_tg[1])

        t_labels[3 * num_test + 2 * i] = 1
        t_labels[3 * num_test + 2 * i + 1] = -1
        img_tg = Img_th.gen_test()
        t_inputs[3 * num_test + 2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[3 * num_test + 2 * i + 1, :, :, :] = ob.representation(
            img_tg[1])

    _epc = 1
    t_inputs = fl.norma_rep(t_inputs)

    acc_list[0, 1:] = ob.test(net, t_inputs, num_test, t_labels)[:-1]
    acc_list[0, 0] = acc_list[0, 1]
    best_acc = np.zeros(np.size(acc_list[0, :]))

    # Training
    weight = np.ones([40, 1])
    for epoch in range(bg_epoch):  # loop over the dataset multiple times
        for i in range(num_batch):
            labels[i], img_tg = Img_L.gen_train()
            inputs[i, :, :, :] = ob.representation(img_tg)

        inputs = fl.norma_rep(inputs)
        inputs = fl.feedforward(inputs,weight)
        inputs = fl.norma_rep(inputs)


        if epoch > 0:
            weight = fl.update_weight(weight,inputs)

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
            t_input_ff = fl.feedforward(t_inputs,weight)
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

def sec_train(show_epc, net_name, bg_epoch=0):
    # input
    num_batch = 16
    Img_L = ob.GenImg(orient='V', loc="L", diff=0)
    Img_R = ob.GenImg(orient='V', loc="R", diff=0)
    Img_DT = ob.GenImg(orient='H', loc="R", diff=0)
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
        t_inputs[2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[2 * i + 1, :, :, :] = ob.representation(img_tg[1])

        t_labels[1 * num_test + 2 * i] = 1
        t_labels[1 * num_test + 2 * i + 1] = -1
        img_tg = Img_DT.gen_test()
        t_inputs[1 * num_test + 2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = ob.representation(
            img_tg[1])

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1
        img_tg = Img_R.gen_test()
        t_inputs[2 * num_test + 2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = ob.representation(
            img_tg[1])

        t_labels[3 * num_test + 2 * i] = 1
        t_labels[3 * num_test + 2 * i + 1] = -1
        img_tg = Img_th.gen_test()
        t_inputs[3 * num_test + 2 * i, :, :, :] = ob.representation(img_tg[0])
        t_inputs[3 * num_test + 2 * i + 1, :, :, :] = ob.representation(
            img_tg[1])

    _epc = 1
    t_inputs = fl.norma_rep(t_inputs)

    acc_list[0, 1:] = ob.test(net, t_inputs, num_test, t_labels)[:-1]
    acc_list[0, 0] = acc_list[0, 1]
    best_acc = np.zeros(np.size(acc_list[0, :]))

    # Training
    weight = np.ones([40, 1])
    for epoch in range(200):  # loop over the dataset multiple times

        for i in range(num_batch):
            labels[i], img_tg = Img_DT.gen_train()
            inputs[i, :, :, :] = ob.representation(img_tg)

        inputs = fl.norma_rep(inputs)
        inputs = fl.feedforward(inputs,weight)
        inputs = fl.norma_rep(inputs)


        if epoch > 20:
            weight = fl.update_weight(weight,inputs)

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
            t_input_ff = fl.feedforward(t_inputs,weight)
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

day = datetime.now().strftime('%Y-%m-%d')
foldername = './' + day
ob.mkdir(foldername)
num_section = 50
num_test = 10
ori = 22.5
loc = 5
show_epoch = 2
print_test = False
net_list = ['s_cc3']
t_data, t_label = GenTestData(_ori=ori)
t_data = fl.norma_rep(t_data)

# pdb.set_trace()

for net_name in net_list:
    for begin_epoch in [20, 80, 140, 200]:
        num_epoch = begin_epoch//show_epoch+1
        AccALL1 = np.zeros([num_section, num_epoch, 5])
        LossALL1 = np.zeros([num_section, num_epoch-1, 1])
        AccALL2 = np.zeros([num_section, 101, 5])
        LossALL2 = np.zeros([num_section, 100, 1])
        for s in range(num_section):
            ### Training
            AccALL1[s, :, :], LossALL1[s, :, :], net, weight = first_train(
                show_epc=show_epoch,
                net_name=net_name,
                bg_epoch=begin_epoch)

            AccALL2[s, :, :], LossALL2[s, :, :], net, weight = sec_train(
                show_epc=show_epoch,
                net_name=net_name,
                bg_epoch=200)

            ### Testing for different location


        name_head = foldername + '/eSpec_'+str(begin_epoch) + net_name
        np.save(name_head + net_name + 'AccALL1.npy', AccALL1)
        np.save(name_head + 'LossALL1.npy', LossALL1)
        np.save(name_head + net_name + 'AccALL2.npy', AccALL2)
        np.save(name_head + 'LossALL2.npy', LossALL2)


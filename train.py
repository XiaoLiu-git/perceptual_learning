import numpy as np
import torch
import torch.optim as optim

from Network import *
from Representation import GenImg, representation


def test(net, inputs, num_batch, labels):
    with torch.no_grad():
        b_x = torch.tensor(inputs, dtype=torch.float32)
        _outputs = net(b_x)
        outputs = _outputs[:3 * num_batch] - _outputs[3 * num_batch:]
        outputs = outputs.squeeze(1)
        acc = ACC(outputs, labels, num_batch)
        print("Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(acc[0] * 100,
                                                                 acc[1] * 100,
                                                                 acc[2] * 100,
                                                                 acc[3] * 100))


def ACC(outputs, labels, num_batch):
    # pdb.set_trace()
    num_test = len(outputs) // num_batch
    acc = np.zeros(num_test + 1)
    total = torch.sign((outputs * labels + 1) / 2)
    for i in range(num_test):
        acc[i] = torch.sum(total[num_batch * i:num_batch * (i + 1)]) / num_batch
    acc[-1] = torch.sum(total) / len(outputs)
    return acc


def np_sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def np_acc(outputs, labels):
    output = outputs.detach().numpy()
    total = np.sign((output * labels + 1) / 2)
    acc = np.sum(total) / len(output)
    return acc


def train(show_epc):
    # input
    num_batch = 16
    Img_L = GenImg(theta=90, loc="L")
    Img_R = GenImg(theta=90, loc="R")
    Img_th = GenImg(theta=180, loc="L")
    inputs = np.zeros([num_batch * 2, 1, 40, 18])
    labels = np.zeros([num_batch])

    # network
    # net = Net_cc()
    # net = Net_cf()
    # net = Net_fc()
    net = Net_ff()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': 1e-3}
    ], lr=1e-3)
    running_loss = 0.0

    num_test = 100
    t_inputs = np.zeros([num_test * 3 * 2, 1, 40, 18])
    t_labels = np.zeros([num_test * 3])
    for i in range(num_test):
        t_inputs[i, :, :, :] = representation(Img_L.gen_reference())
        t_inputs[1 * num_test + i, :, :, :] = representation(
            Img_R.gen_reference())
        t_inputs[2 * num_test + i, :, :, :] = representation(
            Img_th.gen_reference())

        t_labels[i], img_tg = Img_L.gen_theta()
        t_inputs[3 * num_test + i, :, :, :] = representation(img_tg)

        t_labels[1 * num_test + i], img_tg = Img_R.gen_theta()
        t_inputs[4 * num_test + i, :, :, :] = representation(img_tg)

        t_labels[2 * num_test + i], img_tg = Img_th.gen_theta()
        t_inputs[5 * num_test + i, :, :, :] = representation(img_tg)

    test(net, t_inputs, num_test, t_labels)

    # Training
    for epoch in range(200):  # loop over the dataset multiple times

        for i in range(num_batch):
            inputs[i, :, :, :] = representation(Img_L.gen_reference())
            labels[i], img_tg = Img_L.gen_theta()
            inputs[num_batch + i, :, :, :] = representation(img_tg)

        optimizer.zero_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        _outputs = net(b_x)
        outputs = _outputs[:num_batch] - _outputs[num_batch:]
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, torch.FloatTensor((labels + 1) // 2))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch % show_epc == show_epc - 1:  # print every show epoch
            acc = np_acc(outputs, labels)
            print('[%d] loss: %.6f' %
                  (epoch + 1, running_loss / show_epc))
            print('train acc: %.2f %%' % (acc * 100))
            PATH = './net.pth'
            torch.save(net.state_dict(), PATH)
            running_loss = 0.0
            test(net, t_inputs, num_test, t_labels)


train(show_epc=20)

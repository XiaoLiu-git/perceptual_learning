from datetime import datetime
import os

import matplotlib.pyplot as plt

from offline_Hebb import *


def GenTestData(_ori, num=10, loc="L", diff=0):
    num_ori = int(180 / _ori)
    dataset = np.zeros([num* 2 * num_ori, 1, 40, 18])
    label = np.zeros([num* 2 * num_ori])
    for i in range(num_ori):
        ori_i = i * _ori
        Img = GenImg(orient=ori_i, loc=loc, diff=diff)
        for ii in range(num):
            label[i * num * 2 + ii] = 1
            label[i * num * 2 + ii + num] = -1
            img = Img.gen_test()
            # if ii%10==0:
            #     plt.figure(figsize=(3.5,6))
            #     plt.subplot(2,2,1)
            #     plt.imshow(img[0])
            #     plt.subplot(2,2,2)
            #     plt.imshow(representation(img[0]))
            #     plt.subplot(2, 2, 3)
            #     plt.imshow(img[1])
            #     plt.subplot(2, 2, 4)
            #     plt.imshow(representation(img[1]))
            #     plt.show()
            # pdb.set_trace()
            dataset[i * num * 2 + ii, :, :, :] = representation(img[0])
            dataset[i * num * 2 + ii + num, :, :, :] = representation(
                img[1])
    return dataset, label


day = datetime.now().strftime('%Y-%m-%d')
foldername = './' + day
mkdir(foldername)
num_section = 50
num_test = 10
ori = 22.5
loc = 5
AccALL = np.zeros([num_section, 101, 5])
LossALL = np.zeros([num_section, 100, 1])
Acc_test = np.zeros([num_section, int(40 / loc), int(180 / ori)])
print_test = False
net_list = ['s_cc3']
t_data, t_label = GenTestData(_ori=ori)
size_input = t_data.shape
t_data = np.reshape(normalize(np.reshape(t_data, [size_input[0], -1])),
                    size_input)

for net_name in net_list:
    for s in range(num_section):
        AccALL[s, :, :], LossALL[s, :, :], best_net = train(show_epc=2,
                                                          net_name=net_name,
                                                          slow_learning=False)
        for i in range(40//loc):
            Acc_test[s, i, :] = test(best_net, t_data, num_test*2, t_label,
                                     print_test)[:-1]
            t_data = np.roll(t_data, -5, axis=2)

    np.save(foldername+'/'+net_name+'AccALL_1.npy', AccALL)
    np.save(foldername+'/'+net_name+'LossALL_1.npy', LossALL)
    np.save(foldername+'/'+net_name+'Acc_test_1.npy', Acc_test)

import pdb

import numpy as np
import torch

import offline_Hebb as ob


def ExtractFeature(network, layer_name, input_x, num_batch, labels, prt=False):
    features_in_hook = []
    features_out_hook = []

    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None

    for (name, module) in network.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook=hook)

    with torch.no_grad():
        b_x = torch.tensor(input_x, dtype=torch.float32)
        outputs = network(b_x)
        outputs = outputs.squeeze(1)
        acc = ob.ACC(outputs, labels, num_batch)
        if prt:
            print("Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(
                acc[0] * 100,
                acc[1] * 100,
                acc[2] * 100,
                acc[3] * 100,
                acc[4] * 100
            ))
    return acc, features_in_hook, features_out_hook


def GenTestData(_ori, _loc, num=10, diff=0):
    num_ori = int(180 / _ori)
    num_loc = int(400 / _loc)
    dataset = np.zeros([num * 2 * num_ori * num_loc, 1, 40, 18])
    label = np.zeros([num * 2 * num_ori * num_loc])
    for i in range(num_ori * num_loc):
        ori_i = i // num_loc * _ori
        loc_i = i % num_loc
        Img = ob.GenImg(orient=ori_i, loc='L', diff=diff)
        for ii in range(num):
            label[i * num * 2 + ii] = 1
            label[i * num * 2 + ii + num] = -1
            img = Img.gen_test()
            # pdb.set_trace()
            img_ = np.roll(img[0], _loc * loc_i, axis=0)
            dataset[i * num * 2 + ii, :, :, :] = ob.representation(img_)
            img_ = np.roll(img[1], _loc * loc_i, axis=0)
            dataset[i * num * 2 + ii + num, :, :, :] = ob.representation(
                img_)
    return dataset, label


def visualization(network, layer_name):
    ori = 22.5
    loc = 50
    num = 10
    print_tag = False
    t_data, t_label = GenTestData(_ori=ori, _loc=loc)
    size_input = t_data.shape
    t_data = np.reshape(ob.normalize(np.reshape(t_data, [size_input[0], -1])),
                        size_input)
    acc, feature_in, feature_out = ExtractFeature(network, layer_name, t_data,
                                                  num, t_label,
                                                  prt=print_tag)
    # pdb.set_trace()
    return acc, feature_in, feature_out
    ###TODO!!!不知道生成的数据怎么样怎么分析

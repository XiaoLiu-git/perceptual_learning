import pdb

import numpy as np
import offline_Hebb as ob
import representation_test as rt
import matplotlib.pyplot as plt

def update_weight(w, inputs, l_lambda=.1):
    # pdb.set_trace()
    inputs=np.where((inputs > -.5)&(inputs < .5), 0, inputs)
    inputs = np.sign(inputs)
    dw = np.mean(np.mean(inputs[:-1, :, :, :] * inputs[1:, :, :, :], axis=0),
                 axis=-1)
    w = w + l_lambda * dw.T
    # print(dw)
    return w

def norma_rep(represtation):
    size_input = represtation.shape
    represtation = np.reshape(
        normalize(np.reshape(represtation, [size_input[0], -1])), size_input)
    return represtation


def normalize(x):
    mean = np.expand_dims(np.mean(x, axis=1), 1)
    sigma = np.expand_dims(np.std(x, axis=1), 1)
    sigma[sigma == 0] = 1
    nor_x = (x - mean) / sigma
    return nor_x


def feedforward(x, W):
    """
    :param x:size [num * 2 * num_ori, 1, 40, 18]
    :param W: size[1, 18]
    :return: Y: same size as x, Y=x*W
    """
    return x * W


#
# ## figure 3
# inputs = np.zeros([16, 1, 40, 18])
# tests = np.zeros([40, 18])
# labels = np.zeros([16])
# Img_L = ob.GenImg(orient='V', loc="L", diff=0)
# Img_R = ob.GenImg(orient='V', loc="R", diff=0)
# Img_HL = ob.GenImg(orient='H', loc="L", diff=0)
# Img_HR = ob.GenImg(orient='H', loc="R", diff=0)
# test = np.zeros([4, 40, 18])
# _, Img=Img_L.gen_train()
# # rt.show_img(Img)
# test[0,:,:]=ob.representation(Img)
# # rt.show_img(test[0,:,:])
# _, Img = Img_R.gen_train()
# # rt.show_img(Img)
# test[1,:,:]=ob.representation(Img)
# # rt.show_img(test[1,:,:])
# _, Img=Img_HL.gen_train()
# # rt.show_img(Img)
# test[2,:,:]=ob.representation(Img)
# # rt.show_img(test[2,:,:])
# _, Img = Img_HR.gen_train()
# # rt.show_img(Img)
# test[3,:,:]=ob.representation(Img)
# # rt.show_img(test[3,:,:])
# test=norma_rep(test)
# plt.figure()
# for i in range(4):
#     plt.subplot(1, 4, i+1)
#     plt.imshow(test[i,:,:])
#
# plt.show()
#
# weight = np.ones([40,1])
# for epoch in range(200):  # loop over the dataset multiple times
#     for i in range(16):
#         labels[i], img_tg = Img_L.gen_train()
#         activity_tg = ob.representation(img_tg)
#         inputs[i, :, :, :] = activity_tg
#     inputs=norma_rep(inputs)
#
#     weight = update_weight(weight,inputs)
#     if (epoch+1)%50==0:
#         y = feedforward(test, weight)
#         y = norma_rep(y)
#         plt.figure()
#         for i in range(4):
#             plt.subplot(1, 4, i + 1)
#             plt.imshow(y[i, :, :])
#
#         plt.show()
#
#


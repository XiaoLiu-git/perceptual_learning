import pdb

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import numpy as np
from tool_gabor import Vernier, Gabor

plt.style.use("fivethirtyeight")
class GenImg:
    def __init__(self, size=None, orient='H', loc="L", noise_cutout=0.5, diff=2,
                 var_noise=1):
        """

        :param size: [400, 200]
        :param orient: H(horizontal) or V(vertical)
        :param loc: "L" or "R"
        :param noise_cutout: 0 ~ 1
        :param diff: 0-2 Hard-->Easy
        :param var_noise: variation of diff level --> noise: 1
        """

        if size is None:
            size = [400, 200]
        self.w, self.h = size
        self.orient = orient
        self.loc = loc
        self.noise_cutout = noise_cutout
        self.diff = diff
        self.var_n = var_noise
        self.vn = Vernier()

    def gen_reference(self, diff=None):
        if diff is not None:
            self.diff = diff

        self.label = 0
        vernier = self.vn.genVernier([self.w // 2, self.h], self.orient,
                                     self.diff, self.label, self.var_n)
        self.tg = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout
        if self.loc == "L":
            self.tg[:self.w // 2, :] += vernier
        else:
            self.tg[self.w // 2:, :] += vernier
        return self.tg

    def gen_train(self, diff=None):
        if diff is not None:
            self.diff = diff

        self.label = np.sign(np.random.rand(1) - 0.5)
        vernier = self.vn.genVernier([self.w // 2, self.h], self.orient,
                                     self.diff, self.label, self.var_n)
        self.tg = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout
        if self.loc == "L":
            self.tg[:self.w // 2, :] += vernier
        else:
            self.tg[self.w // 2:, :] += vernier
        return self.label, self.tg

    def gen_test(self, diff=None):
        if diff is not None:
            self.diff = diff

        self.label = 1
        vernier_p = self.vn.genVernier([self.w // 2, self.h], self.orient,
                                     self.diff, self.label, self.var_n)
        self.tg_p = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout
        self.label = -1
        vernier_n = self.vn.genVernier([self.w // 2, self.h], self.orient,
                                       self.diff, self.label, self.var_n)
        self.tg_n = (np.random.rand(self.w,
                                    self.h) - 0.5) * 2 * self.noise_cutout
        if self.loc == "L":
            self.tg_p[:self.w // 2, :] += vernier_p
            self.tg_n[:self.w // 2, :] += vernier_n
        else:
            self.tg_p[self.w // 2:, :] += vernier_p
            self.tg_n[:self.w // 2, :] += vernier_n
        return self.tg_p, self.tg_n

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    # plt.show(block=False)
    # plt.pause(5)
    # plt.close()


def representation(img, num_x=40, num_theta=18):
    [w, h] = img.shape
    activity = np.zeros([num_x, num_theta])
    func_size = [w // 4, h, num_theta]
    basis_gabor = np.zeros(func_size)
    gb = Gabor(sigma=30, freq=.01)
    for theta in range(num_theta):
        # pdb.set_trace()
        basis_gabor[:, :, theta] = gb.genGabor(func_size[:-1],
                                               theta * 180 / num_theta)
        # show_img(basis_gabor[:,:,theta])
    for x in range(num_x):
        center = w * x // num_x
        # pdb.set_trace()
        img_cut = np.roll(img, center, axis=0)[:func_size[0], :]
        # show_img(img_cut)
        for theta in range(num_theta):
            # pdb.set_trace()
            activity[x, theta] = (img_cut * basis_gabor[:, :, theta]).sum()
    return activity


# genimg = GenImg(orient='V', loc="R", diff=2)
#
# label, img_tg = genimg.gen_train()
#
# show_img(img_tg.T)
#
# activity_tg = representation(img_tg)
# show_img(activity_tg)
#
# print(label)
#
# genimg = GenImg(orient='V', loc="R", diff=2)
#
# label, img_tg = genimg.gen_train()
#
# show_img(img_tg.T)
#
# activity_tg = representation(img_tg)
# show_img(activity_tg)
# print(label)
#
# genimg = GenImg(orient='V', loc="R", diff=1)
#
# label, img_tg = genimg.gen_train()
#
# show_img(img_tg.T)
#
# activity_tg = representation(img_tg)
# show_img(activity_tg)
# print(label)
#
# genimg = GenImg(orient='V', loc="R", diff=1)
#
# label, img_tg = genimg.gen_theta()
#
# show_img(img_tg.T)
#
# activity_tg = representation(img_tg)
# show_img(activity_tg)
# print(label)

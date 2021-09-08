import matplotlib.pyplot as plt
import numpy as np
import pdb
from tool_gabor import Gabor


class GenImg:
    def __init__(self, size=None, theta=0, loc="L", noise_cutout=0.1, diff=10):
        """

        :param size: [400, 200]
        :param theta: 0 ~ 360
        :param loc: "L" or "R"
        :param noise_cutout: 0 ~ 1
        """

        if size is None:
            size = [400, 200]
        self.w, self.h = size
        self.theta = theta
        self.loc = loc
        self.noise_cutout = noise_cutout
        self.diff = diff
        self.gb = Gabor()

    def gen_reference(self):
        gabor = self.gb.genGabor([self.w // 2, self.h], self.theta)

        self.ref = (np.random.rand(self.w,
                                   self.h) - 0.5) * 2 * self.noise_cutout
        if self.loc == "L":
            self.ref[:self.w // 2, :] += gabor
        else:
            self.ref[self.w // 2:, :] += gabor
        return self.ref

    def gen_theta(self):
        self.label = np.sign(np.random.rand(1) - 0.5)
        theta_tg = self.theta + self.label * self.diff
        gabor = self.gb.genGabor([self.w // 2, self.h], theta_tg)
        self.tg = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout
        if self.loc == "L":
            self.tg[:self.w // 2, :] += gabor
        else:
            self.tg[self.w // 2:, :] += gabor
        return self.label, self.tg

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def representation(img, num_x=40, num_theta=18):

    [w, h] = img.shape
    activity = np.zeros([num_x, num_theta])
    func_size = [w//4, h, num_theta]
    basis_gabor = np.zeros(func_size)
    gb = Gabor(sigma=30,freq=.01)
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

# genimg = GenImg(theta=90)
# img_ref = genimg.gen_reference()
# show_img(img_ref.T)
# label, img_tg = genimg.gen_theta()
# show_img(img_tg.T)
# activity_ref = representation(img_ref)
# show_img(activity_ref)
# activity_tg = representation(img_tg)
# show_img(activity_tg)
# show_img(activity_ref-activity_tg)
# print(label)
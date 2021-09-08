import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import numpy as np


def grating(X, Y, params):
    """
    GRATING --  Sinusoidal grating (grayscale image).
    %
    %  G = grating(X,Y,params)
    %
    %  X and Y are matrices produced by MESHGRID (use integers=pixels).
    %  PARAMS is a struct w/ fields 'amplit', 'freq', 'orient', and 'phase'.
    %  The function GABOR_PARAMS supplies default parameters.
    %  G is a matrix of luminance values.  size(G)==size(X)==size(Y)
    %
    %  Example:
    %    x=[-100:+100] ; y=[-120:+120] ; [X,Y] = meshgrid(x,y) ;
    %    params = gabor_params ; params.orient = 15*pi/180 ;
    %    G = grating(X,Y,params) ;
    %    imagesc1(x,y,G) ;

    """
    A = params["amplit"]
    omega = 2 * np.pi * params["freq"]
    theta = params["orient"]
    phi = params["phase"]
    slant = X * (omega * np.cos(theta)) + Y * (
            omega * np.sin(theta))  # cf. function SLANT
    G = A * np.cos(slant + phi)
    return G


def gabor(X, Y, params):
    """
    %GABOR  --  Sinusoidal grating under a Gaussian envelope.
    %
    %  G = gabor(X,Y,params)
    %
    %  X and Y are matrices produced by MESHGRID (use integers=pixels).
    %  PARAMS is a struct with fields 'amplit', 'freq', 'orient', 'phase',
    %  and 'sigma'. The function GABOR_PARAMS supplies default parameters.
    %  G is a matrix of luminance values.  size(G)==size(X)==size(Y)
    %
    %  Example:
    %    x=[-100:+100] ; y=[-120:+120] ; [X,Y] = meshgrid(x,y) ;
    %    params = gabor_params ; params.orient = 60*pi/180 ;
    %    G = gabor(X,Y,params) ;
    %    imagesc1(x,y,G) ;

    """

    sigmasq = params["sigma"] ** 2

    Gaussian = np.exp(-(X ** 2 + Y ** 2) / (2 * sigmasq))
    Grating = grating(X, Y, params)
    G = Gaussian * Grating
    return G


class Gabor:
    def __init__(self, sigma=25, freq=0.2):
        self.params = {
            "amplit": 0.5,  # amplitude [luminance units], min=-A,max=+A
            "freq": freq,  # spatial frequency [cycles/pixel]
            "orient": 0,  # orientation [radians]
            "phase": 1.5708,  # phase [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
        }

    def genGabor(self, size, orient):
        """
        :param size:
        :param orient: 0~360
        :return: G
        """
        x = np.arange(-size[1] // 2, size[1] // 2)
        y = np.arange(-size[0] // 2, size[0] // 2)
        X, Y = np.meshgrid(x, y)
        self.params["orient"] = orient * np.pi / 180
        self.G = gabor(X, Y, self.params)
        return self.G

    def showGabor(self):
        plt.figure()
        plt.imshow(self.G)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()


class Vernier:
    def __init__(self, sigma=9, freq=0.06):
        self.params = {
            "amplit": 0.5,  # amplitude [luminance units], min=-A,max=+A
            "freq": freq,  # spatial frequency [cycles/pixel]
            "orient": 0,  # orientation [radians]
            "phase": 0,  # phase [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
            "diff_level": [1, 3, 5]
        }

    def genVernier(self, size, orient, diff, label):
        """
        :param size:
        :param orient: H(horizontal) or V(vertical)
        :param diff: 0,1,2 (easy medium hard)
        :param label: +1 or -1
        :return: G
        """
        # generate Gabor
        if orient == "H":
            _orient = 0
        else:
            _orient = 90
        x = np.arange(-size[1] // 4, size[1] // 4)
        y = np.arange(-size[0] // 4, size[0] // 4)
        X, Y = np.meshgrid(x, y)
        self.params["orient"] = _orient * np.pi / 180
        G = gabor(X, Y, self.params)

        # arrange Gabor into Vernier
        jitter = self.params["diff_level"][diff] * label
        self.V = np.zeros(size)
        if orient == "H":
            self.V[0:size[0] // 2, size[1] // 4:size[1] * 3 // 4] = G
            self.V[size[0] // 2:,
            size[1] // 4 + jitter: size[1] * 3 // 4 + jitter] = G
        else:
            self.V[size[0] // 4:size[0] * 3 // 4, 0:size[1] // 2] = G
            self.V[size[0] // 4 + jitter: size[0] * 3 // 4 + jitter,
            size[1] // 2:] = G

        return self.V

    def show(self):
        plt.figure()
        plt.imshow(self.V)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()




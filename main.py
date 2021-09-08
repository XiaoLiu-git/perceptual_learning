import matplotlib.pyplot as plt
import numpy as np

len_x = 40
len_th = 18
len_M = 40 * 18
lr = 0.01
tua = 10

stm1_L = np.zeros([len_x, len_th])
stm2_L = np.zeros([len_x, len_th])
stm1_R = np.zeros([len_x, len_th])
stm2_R = np.zeros([len_x, len_th])

# 生成标签 放到训练CNN的地方去
np.floor((np.random.rand(100) - 0.5)) * 2 + 1

# 生成 firing rate 表征
x0 = np.arange(-5, 5, 0.5)
th0 = np.arange(-9, 9, 1)
fr_th = np.exp(-(th0 / 2) ** 2)
fr_x = np.exp(-(x0 / 2) ** 2)

thta = np.sign((np.random.rand(1) - 0.5) / 0.5) * np.random.randint(4)
thta = thta.astype(int)
cts = (np.random.rand(1) - 0.5) / 0.5 * 0.1
for i in range(20):
    for j in range(18):
        # pdb.set_trace()
        stm1_L[i, j] = fr_th[j] * fr_x[i]
        stm2_L[i, np.mod(j + thta, 18)] = fr_th[j] * fr_x[i]

        stm1_R[i + 20, j] = fr_th[j] * fr_x[i]
        stm2_R[i + 20, j] = (1 - cts) * fr_th[j] * fr_x[i]

# 表征中加入noise
STM = {}
STM['stm1_L'] = stm1_L + np.reshape(0.05 * np.random.rand(len_x * len_th),
                                    (len_x, len_th))
STM['stm2_L'] = stm2_L + np.reshape(0.05 * np.random.rand(len_x * len_th),
                                    (len_x, len_th))
STM['stm1_R'] = stm1_R + np.reshape(0.05 * np.random.rand(len_x * len_th),
                                    (len_x, len_th))
STM['stm2_R'] = stm2_R + np.reshape(0.05 * np.random.rand(len_x * len_th),
                                    (len_x, len_th))
fig = plt.figure(1)
for i, name in enumerate(STM.keys()):
    plt.subplot(1, 4, i + 1)
    plt.imshow(STM[name])
fig.text(0.5, 0.04, 'Theta ${\\theta} $', ha='center', va='center')
fig.text(0.06, 0.5, 'Location $x$', ha='center', va='center',
         rotation='vertical')
plt.show()

# similarity preserving
W = np.eye((len_x * len_th))
M = np.ones((len_M, len_M)) - np.eye(len_M)
simpre = np.zeros(len_x * len_th)

for t in range(20):
    stm1_L = np.zeros([len_x, len_th])
    stm2_L = np.zeros([len_x, len_th])
    for i in range(20):
        for j in range(18):
            # pdb.set_trace()
            stm1_L[i, j] = fr_th[j] * fr_x[i]
            stm2_L[i, np.mod(j + thta, 18)] = fr_th[j] * fr_x[i]

    fig = plt.figure()
    x1t = np.reshape(
        stm1_L + np.reshape(0.05 * np.random.rand(len_x * len_th),
                            (len_x, len_th)), (40 * 18, 1))
    y1t = np.dot(np.dot(np.linalg.inv(M), W), x1t)
    W = W + 2 * lr * (np.dot(y1t, x1t.T) - W)
    M = M + lr / tua * (np.dot(y1t, y1t.T) - M)
    plt.subplot(1,4,1)
    Y1t=np.reshape(y1t, (40, 18))
    plt.imshow(Y1t)
    x2t = np.reshape(
        stm2_L + np.reshape(0.05 * np.random.rand(len_x * len_th),
                            (len_x, len_th)), (40 * 18, 1))
    y2t = np.dot(np.dot(np.linalg.inv(M), W), x2t)
    # pdb.set_trace()
    Y2t = np.reshape(y2t, (40, 18))
    W = W + 2 * lr * (np.dot(y2t, x2t.T) - W)
    M = M + lr / tua * (np.dot(y2t, y2t.T) - M)
    plt.subplot(1, 4, 2)
    plt.imshow(Y2t)
    plt.subplot(1,4,3)
    X1t=np.reshape(x1t, (40, 18))
    X2t = np.reshape(x2t, (40, 18))
    plt.imshow(np.reshape(x1t*x2t, (40, 18)))
    plt.subplot(1,4,4)
    plt.imshow(np.reshape(y1t*y2t, (40, 18)))
    plt.show()

for t in range(20):
    stm1_R = np.zeros([len_x, len_th])
    stm2_R = np.zeros([len_x, len_th])
    for i in range(20):
        for j in range(18):
            stm1_R[i + 20, j] = fr_th[j] * fr_x[i]
            stm2_R[i + 20, j] = (1 - cts) * fr_th[j] * fr_x[i]

    fig = plt.figure()
    x1t = np.reshape(
        stm1_R + np.reshape(0.05 * np.random.rand(len_x * len_th),
                            (len_x, len_th)), (40 * 18, 1))
    y1t = np.dot(np.dot(np.linalg.inv(M), W), x1t)
    W = W + 2 * lr * (np.dot(y1t, x1t.T) - W)
    M = M + lr / tua * (np.dot(y1t, y1t.T) - M)
    plt.subplot(1,4,1)
    Y1t=np.reshape(y1t, (40, 18))
    plt.imshow(Y1t)
    x2t = np.reshape(
        stm2_R + np.reshape(0.05 * np.random.rand(len_x * len_th),
                            (len_x, len_th)), (40 * 18, 1))
    y2t = np.dot(np.dot(np.linalg.inv(M), W), x2t)
    # pdb.set_trace()
    Y2t = np.reshape(y2t, (40, 18))
    W = W + 2 * lr * (np.dot(y2t, x2t.T) - W)
    M = M + lr / tua * (np.dot(y2t, y2t.T) - M)
    plt.subplot(1, 4, 2)
    plt.imshow(Y2t)
    plt.subplot(1,4,3)
    X1t=np.reshape(x1t, (40, 18))
    X2t = np.reshape(x2t, (40, 18))
    plt.imshow(np.reshape(x1t*x2t, (40, 18)))
    plt.subplot(1,4,4)
    plt.imshow(np.reshape(y1t*y2t, (40, 18)))
    plt.show()
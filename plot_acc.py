import pdb

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = [
    'Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['svg.fonttype'] = 'none'
plt.style.available
plt.style.use('seaborn')
a = np.load('./2022-12-04/task_modules_cc3Acc_test.npy')
c = np.mean(a, axis=0) * 100
# plt.figure()
# plt.plot(c[:, 0, 2], label='trained', linewidth=4)
# plt.plot(c[:, 0, 2], label='ori1_loc1', linewidth=4)
# plt.plot(c[:, 4, 6], label='ori2_loc2', linewidth=4)
# plt.plot(c[:, 4, 2], label='ori1_loc2', linewidth=4)
# plt.plot(c[:, 0, 6], label='ori2_loc1', linewidth=4)
# plt.ylabel("Accuracy %", fontsize=18)
# plt.xlabel("No. epoch", fontsize=18)
# plt.legend(prop={'size': 18})
# plt.show()
#
# a = np.load('./2022-04-13/db_af_20_005s_cf3s_cf3AccALL.npy')
a = np.load('./2022-12-03/eSpec_20s_cc3s_cc3AccALL1.npy')
Acc = np.mean(a, axis=0) * 100
pdb.set_trace()
plt.figure()
plt.plot(np.arange(0, 202, 2), Acc, linewidth=4,
         label=["train", "ori1_loc1", "ori2_loc2", "ori1_loc2",
                "ori2_loc1"])
plt.ylabel("Accuracy %", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.legend(prop={'size': 18})
# plt.show()

a = np.load('./2022-04-13/db_af_20_005s_cf3LossALL.npy')
Acc = np.mean(a, axis=0)
plt.figure()
plt.plot(np.arange(0, 200, 2), Acc, linewidth=4, )
plt.ylabel("Loss", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.show()

fig, ax = plt.subplots()
a = np.load('./2022-12-09/eSpec_20s_cc3s_cc3AccALL1.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 22, 2), Acc[:, 1], linewidth=4, label='Train 20')
a = np.load('./2022-12-09/eSpec_80s_cc3s_cc3AccALL1.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 82, 2), Acc[:, 1], linewidth=4, label='Train 80')
a = np.load('./2022-12-09/eSpec_140s_cc3s_cc3AccALL1.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 142, 2), Acc[:, 1], linewidth=4, label='Train 140')
a = np.load('./2022-12-09/eSpec_200s_cc3s_cc3AccALL1.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 202, 2), Acc[:, 1], linewidth=4, label='Train 200')
plt.ylabel("Accuracy %", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.legend(prop={'size': 18})

fig, ax = plt.subplots()
a = np.load('./2022-12-09/eSpec_20s_cc3s_cc3AccALL2.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 202, 2), Acc[:, 2], linewidth=4, label='Train 20')
a = np.load('./2022-12-09/eSpec_80s_cc3s_cc3AccALL2.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 202, 2), Acc[:, 2], linewidth=4, label='Train 40')
a = np.load('./2022-12-09/eSpec_140s_cc3s_cc3AccALL2.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 202, 2), Acc[:, 2], linewidth=4, label='Train 140')
a = np.load('./2022-12-09/eSpec_200s_cc3s_cc3AccALL2.npy')
Acc = np.mean(a, axis=0) * 100
ax.plot(np.arange(0, 202, 2), Acc[:, 2], linewidth=4, label='Train 200')
plt.ylabel("Accuracy %", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.legend(prop={'size': 18})

a = np.load('./2022-12-04/task_modules_cc3s_cc3AccALL.npy')
c = np.mean(a, axis=0) * 100
plt.figure()
plt.plot(np.arange(0, 202, 2), c, linewidth=4,
         label=['train', 'test', 'ori', 'loc5', 'loc10', 'loc15', 'loc20',
                'loc25', 'loc30', 'loc35'])
plt.ylabel("Accuracy %", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.legend(prop={'size': 18})

a = np.load('./2022-12-04/gen_0051s_cc3AccALL.npy')
c = np.mean(a, axis=0) * 100
plt.figure()
plt.plot(np.arange(0, 202, 2), c, linewidth=4,
         label=['train', 'test', 'ori', 'loc5', 'loc10', 'loc15', 'loc20',
                'loc25', 'loc30', 'loc35'])
plt.ylabel("Accuracy %", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.legend(prop={'size': 18})

a = np.load('./2022-12-04/gen_0051s_cc3LossALL.npy')
Acc = np.mean(a, axis=0)
plt.figure()
plt.plot(np.arange(0, 200, 2), Acc, linewidth=4, )
plt.ylabel("Loss", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.show()
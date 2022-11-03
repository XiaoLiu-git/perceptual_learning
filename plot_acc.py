import matplotlib.pyplot as plt
import numpy as np
import pdb


plt.rcParams['font.sans-serif'] = [
    'Times New Roman']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['svg.fonttype'] = 'none'
plt.style.available
plt.style.use('seaborn')
a = np.load('./2022-04-13/db_af_20_005s_cf3Acc_testdb.npy')
c = np.mean(a, axis=0) * 100
plt.figure()
plt.plot(c[:, 0, 2], label='trained', linewidth=4)
plt.plot(c[:, 0, 2], label='ori1_loc1', linewidth=4)
plt.plot(c[:, 4, 6], label='ori2_loc2', linewidth=4)
plt.plot(c[:, 4, 2], label='ori1_loc2', linewidth=4)
plt.plot(c[:, 0, 6], label='ori2_loc1', linewidth=4)
plt.ylabel("Accuracy %", fontsize=18)
plt.xlabel("No. epoch", fontsize=18)
plt.legend(prop={'size': 18})
# plt.show()

a = np.load('./2022-04-13/db_af_20_005s_cf3s_cf3AccALL.npy')
# a = np.load('./2022-03-10/learning005s_cc3AccALL.npy')
Acc = np.mean(a, axis=0) * 100
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

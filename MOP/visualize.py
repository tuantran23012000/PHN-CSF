import numpy as np
from matplotlib import pyplot as plt
def get_val(a):
    tmp = []
    tmp.append(a[0])
    tmp.append(a[3])
    tmp.append(a[6])
    tmp.append(a[9])
    return tmp
# med_mlp = get_val(np.load("med_mlp.npy"))
# param_mlp = get_val(np.load("param_mlp.npy"))
# med_t = get_val(np.load("med_t.npy"))
# param_t = get_val(np.load("param_t.npy"))
# med_t_position = get_val(np.load("med_t_position.npy"))
# param_t_position = get_val(np.load("param_t_position.npy"))
example = 'ex3_'
med_mlp = np.load(example+"med_mlp.npy")
# med_mlp1 = np.load(example+"med_mlp.npy")
# med_mlp = np.hstack((med_mlp,med_mlp1))
print(med_mlp)
param_mlp = np.load(example+"param_mlp.npy")
# param_mlp1 = np.load(example+"param_mlp.npy")
# param_mlp = np.hstack((param_mlp,param_mlp1))
med_t = np.load(example+"med_trans.npy")
print(med_t)
param_t = np.load(example+"param_trans.npy")
# med_t_position = np.load(example+"med_trans_posi.npy")
# print(med_t_position)
# param_t_position = np.load(example+"param_trans_posi.npy")
fig, ax = plt.subplots()
ax.plot(param_mlp,med_mlp, '-.',label = 'MLP',linewidth=1)
ax.plot(param_t[:25],med_t[:25],'--',label = 'Transfomer',linewidth=1)
# ax.plot(param_t_position[:20],med_t_position[:20],':',label = 'Transfomer_position',linewidth=1)
# ax.scatter(param_mlp,med_mlp,s = 500, marker='o',label = 'MLP')
# ax.scatter(param_t,med_t,s = 500,marker='o',label = 'Transfomer')
# ax.scatter(param_t_position,med_t_position,s = 500,marker='o',label = 'Transfomer_position')
ax.set_xlabel('Params',fontsize=12)
ax.set_ylabel('MED',fontsize=12)
ax.legend()

plt.savefig("ex1.jpg")

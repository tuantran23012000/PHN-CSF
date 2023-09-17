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
med_mlp = np.load("med_mlp.npy")
param_mlp = np.load("param_mlp.npy")
med_t = np.load("med_t.npy")
param_t = np.load("param_t.npy")
med_t_position = np.load("med_t_position.npy")
param_t_position = np.load("param_t_position.npy")
fig, ax = plt.subplots()
ax.plot(param_mlp,med_mlp, '-.',label = 'MLP',linewidth=3)
ax.plot(param_t,med_t,'--',label = 'Transfomer',linewidth=3)
ax.plot(param_t_position,med_t_position,':',label = 'Transfomer_position',linewidth=3)
# ax.scatter(param_mlp,med_mlp,s = 500, marker='o',label = 'MLP')
# ax.scatter(param_t,med_t,s = 500,marker='o',label = 'Transfomer')
# ax.scatter(param_t_position,med_t_position,s = 500,marker='o',label = 'Transfomer_position')
ax.set_xlabel('Params',fontsize=15)
ax.set_ylabel('MED',fontsize=15)
ax.legend()

plt.savefig("test.jpg")

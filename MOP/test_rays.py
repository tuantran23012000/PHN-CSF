import numpy as np
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from tools.utils import circle_points, sample_vec

def get_rays(n_obj,num_ray_init):
    contexts = np.array(sample_vec(n_obj,num_ray_init))
    tmp = []
    for r in contexts:
        flag = True
        for i in r:
            if i <=0.16:
                flag = False
                break
        if flag:

            tmp.append(r)
    contexts = np.array(tmp)
    return contexts
# def get_ref_dirs(n_obj):
#     if n_obj == 2:
#         ref_dirs = UniformReferenceDirectionFactory(2, n_points=100).do()
#     elif n_obj == 3:
#         #ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=15).do()
#         ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=16).do()
#         tmp = []
#         for r in ref_dirs:
#             if r.all() != 0:
#                 tmp.append(r)
#         ref_dirs = np.array(tmp)
#     return ref_dirs
n = 3
num_ray_init = 25
ref_dirs = get_rays(n,num_ray_init)
print(ref_dirs.shape)
# print(ref_dirs)
np.save("test_rays_3d.npy",ref_dirs)
# n = 2
# ref_dirs = get_ref_dirs(2)
# print(ref_dirs.shape)
# np.save("test_rays_2d.npy",ref_dirs)

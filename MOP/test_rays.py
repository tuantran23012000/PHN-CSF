import numpy as np
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
from tools.utils import circle_points, sample_vec

def get_ref_dirs(n_obj,num_ray_init):
    if n_obj == 2:
        #ref_dirs = UniformReferenceDirectionFactory(2, n_points=num_ray_init).do()
        ref_dirs = np.array(sample_vec(n_obj,num_ray_init))
    else:
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
        ref_dirs = np.array(tmp)
    return ref_dirs
n = 3
ref_dirs = get_ref_dirs(n,25)
print(ref_dirs.shape)
# print(ref_dirs)
np.save("test_rays_3d.npy",ref_dirs)
n = 2
ref_dirs = get_ref_dirs(n,100)
print(ref_dirs.shape)
np.save("test_rays_2d.npy",ref_dirs)

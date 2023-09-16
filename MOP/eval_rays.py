import numpy as np
from pymoo.util.reference_direction import UniformReferenceDirectionFactory
def get_ref_dirs(n_obj,n_points):
    if n_obj == 2:
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=n_points).do()
    elif n_obj == 3:
        #ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=15).do()
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=n_points).do()
        tmp = []
        for r in ref_dirs:
            if r.all() != 0:
                tmp.append(r)
        ref_dirs = np.array(tmp)
    return ref_dirs
n = 3
ref_dirs = get_ref_dirs(3,33)
print(ref_dirs.shape)
np.save("test_rays_3d.npy",ref_dirs)
n = 2
ref_dirs = get_ref_dirs(2,500)
print(ref_dirs.shape)
np.save("test_rays_2d.npy",ref_dirs)

import numpy as np
import random
import torch
from numpy import save, load
from pymoo.factory import get_reference_directions
device = torch.device(f"cuda:0" if torch.cuda.is_available() and not False else "cpu")
def get_test_rays():
    """Create 100 rays for evaluation. Not pretty but does the trick"""
    test_rays = get_reference_directions("das-dennis", 40, n_partitions=7).astype(
        np.float32
    )
    
    test_rays = test_rays[[(r > 0).all() for r in test_rays]][5:-5:2]
    #logging.info(f"initialize {len(test_rays)} test rays")
    return np.array(test_rays)
def sample_vec(n,m):
    vector = [0]*n
    unit = np.linspace(0, 1, m)
    rays = []
    def sample(i, sum):
        if i == n-1:
            vector[i] = 1-sum
            rays.append(vector.copy())
            return vector
        for value in unit:
            if value > 1-sum:
                break
            else:
                vector[i] = value
                sample(i+1, sum+value)
    sample(0,0)
    rays = np.array(rays)
    rays = rays[[(r > 0).all() for r in rays]]
    return rays
def get_ray_test(alpha,num_tasks,m):
                # if alpha > 0:
    rays = []
    check = [alpha]*num_tasks
    for i in range(m):
        ray = np.random.dirichlet(tuple(check), 1).astype(np.float32).flatten()
        #print(ray.sum())
        rays.append(ray.tolist())
    return rays
alpha = 0.6
test_ray = get_ray_test(alpha,40,25)
#test_ray = get_test_rays()
#test_ray = sample_vec(40,8)
print(np.array(test_ray))
print(np.array(test_ray).shape)
save('test_rays.npy', test_ray)
data = load('test_rays.npy')
# print the array
print(data.shape)
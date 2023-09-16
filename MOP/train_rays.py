import torch
import os
import numpy as np

def get_ray(n,alpha_r,num_ray):
    rays = []
    for i in range(num_ray):
        if n == 2:
            ray = np.random.dirichlet((alpha_r, alpha_r), 1).astype(np.float32)[0].tolist()
        else:
            ray = np.random.dirichlet((alpha_r, alpha_r,alpha_r), 1).astype(np.float32)[0].tolist()
        rays.append(ray)
    rays_sort = np.array(rays)
    # rays_train = torch.from_numpy(rays_sort).float()
    # train_dt = torch.utils.data.TensorDataset(rays_train)
    return rays_sort
if __name__ == "__main__":
    train_dt = get_ray(2,0.6,20000)
    print(train_dt.shape)
    np.save("train_rays_2d.npy",train_dt)
    train_dt = get_ray(3,0.6,20000)
    print(train_dt.shape)
    np.save("train_rays_3d.npy",train_dt)
import numpy as np
rays = []
for i in range(10000):
    ray = np.random.dirichlet((0.6, 0.6), 1).astype(np.float32).tolist()
    rays.append(ray)
rays = np.array(rays).reshape(10000,2)
print(rays.shape)
np.save("eval_rays.npy",rays)

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import zarr

z1 = zarr.open('../data/in/czii-cryo-et-object-identification/test/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/denoised.zarr', mode='r')
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
scans = np.array(z1[2])
norm = Normalize(vmin=scans.min(), vmax=scans.max())
# ax.set_box_aspect([1,1,1])
# ax.set_xlim3d([-0.3, 0.3])
# ax.set_ylim3d([-0.3, 0.3])
# ax.set_zlim3d([-0.3, 0.3])
z1_list = np.array(z1[0]).flatten()
print(scans.shape)
i = 0
print(range(scans.shape[0])[-1])
for z in range(scans.shape[0]-1):
    for x in range(scans.shape[1]-1):
        if x == 0:
            continue
        for y in range(scans.shape[2]-1):
            print(x, y, z)

            ax.scatter(x, y, z, color=cm.viridis((scans[x][y][z])))
            i += 1
            break

plt.show()
print(i)
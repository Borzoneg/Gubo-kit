from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import zarr
import open3d as o3d

z1 = zarr.open('../data/in/czii-cryo-et-object-identification/test/static/ExperimentRuns/TS_6_4/VoxelSpacing10.000/denoised.zarr', mode='r')

scans = np.array(z1[2])
norm = Normalize(vmin=scans.min(), vmax=scans.max())

point_cloud = o3d.geometry.PointCloud()
points = []
colors = []
c_values = [np.array(cm.viridis(scans[0][0][0])[:3])]
for z in range(scans.shape[0]):
    for x in range(scans.shape[1]):
        for y in range(scans.shape[2]):
            points.append(np.array((z, x, y)))
            color = np.array(cm.viridis(norm(scans[z][x][y]))[:3])
            # for value in c_values:
                # if np.linalg.norm(value - color) != 0:
                #     c_values.append(color)
                #     print(color)
            colors.append(color)
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([point_cloud])
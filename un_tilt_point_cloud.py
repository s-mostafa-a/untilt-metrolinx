import os

import numpy as np

import open3d as o3d

ROTATION_MATRIX = np.array([[0.99648296, 0.03442719, -0.07639682],
                            [-0.01334566, 0.9652704, 0.26091177],
                            [0.08272604, -0.25897457, 0.96233496]])
# you may want to change these variables
SOURCE_DIRECTORY = './data/point_cloud/source'
OUTPUT_DIRECTORY = './data/point_cloud/output'


def rotate_csv(file_name, visualize=False):
    path_to_source_point_cloud = os.path.join(SOURCE_DIRECTORY, file_name)
    path_to_result_point_cloud = os.path.join(OUTPUT_DIRECTORY, file_name)
    points_with_i = np.genfromtxt(path_to_source_point_cloud, dtype=None, delimiter=',', skip_header=0)
    points = points_with_i[:, :3]
    rotated_points_transpose = ROTATION_MATRIX @ points.T
    rotated_points_transpose = rotated_points_transpose * np.array([[-1], [-1], [1]])
    rotated_points = rotated_points_transpose.T
    points_with_i[:, :3] = rotated_points
    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_with_i[:, :3])
        o3d.visualization.draw_geometries([pcd])
    np.savetxt(path_to_result_point_cloud, points_with_i, delimiter=',')


if __name__ == '__main__':
    for filename in os.listdir(SOURCE_DIRECTORY):
        if filename.endswith(".csv"):
            rotate_csv(file_name=filename, visualize=False)
        else:
            continue

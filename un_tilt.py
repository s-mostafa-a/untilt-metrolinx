import numpy as np
# uncomment all the comment to visualize the point cloud
# import open3d as o3d

ROTATION_MATRIX = np.array([[0.99648296, 0.03442719, -0.07639682],
                            [-0.01334566, 0.9652704, 0.26091177],
                            [0.08272604, -0.25897457, 0.96233496]])


def rotate_csv(path_to_point_cloud, out_put_directory):
    # pcd = o3d.geometry.PointCloud()
    points_with_i = np.genfromtxt(path_to_point_cloud, dtype=None, delimiter=',', skip_header=0)
    points = points_with_i[:, :3]
    rotated_points_transpose = ROTATION_MATRIX @ points.T
    rotated_points_transpose = rotated_points_transpose * np.array([[-1], [-1], [1]])
    rotated_points = rotated_points_transpose.T
    points_with_i[:, :3] = rotated_points
    # pcd.points = o3d.utility.Vector3dVector(points_with_i[:, :3])
    # o3d.visualization.draw_geometries([pcd])
    file_name = path_to_point_cloud.split('/')[-1]
    np.savetxt(f'{out_put_directory}/{file_name}', points_with_i, delimiter=',')


if __name__ == '__main__':

    rotate_csv('./data/source/xyzi_m1255_1187.csv', './data/output')

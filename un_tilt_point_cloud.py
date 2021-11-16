import os
import numpy as np

from utils import rotate_points

# you may want to change these variables
INPUT_DIRECTORY = './not_upload_data/point_cloud/input'
OUTPUT_DIRECTORY = './not_upload_data/point_cloud/output'


def rotate_csv(file_name):
    path_to_source_point_cloud = os.path.join(INPUT_DIRECTORY, file_name)
    path_to_result_point_cloud = os.path.join(OUTPUT_DIRECTORY, file_name)
    points_with_i = np.genfromtxt(path_to_source_point_cloud, dtype=None, delimiter=',',
                                  skip_header=0)
    points_with_i[:, :3] = rotate_points(points_with_i[:, :3])
    np.savetxt(path_to_result_point_cloud, points_with_i, delimiter=',')


if __name__ == '__main__':
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".csv"):
            rotate_csv(file_name=filename)
        else:
            continue

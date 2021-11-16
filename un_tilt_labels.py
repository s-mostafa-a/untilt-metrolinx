import os
import numpy as np
from jacob_land import visualize_utils as JV
from jacob_land import utils as JU
from utils import rotate_points

# you may want to change these variables
INPUT_DIRECTORY = './data/labels/input'
OUTPUT_DIRECTORY = './data/labels/output'


def center_to_eight_corners(x, y, z, height, width, length):
    return np.array([[x + length / 2, y, z + width / 2],
                     [x + length / 2, y, z - width / 2],
                     [x - length / 2, y, z + width / 2],
                     [x - length / 2, y, z - width / 2],
                     [x + length / 2, y - height, z + width / 2],
                     [x + length / 2, y - height, z - width / 2],
                     [x - length / 2, y - height, z + width / 2],
                     [x - length / 2, y - height, z - width / 2]])


def rotate_txt(file_name):
    path_to_source_labels = os.path.join(INPUT_DIRECTORY, file_name)
    path_to_result_labels = os.path.join(OUTPUT_DIRECTORY, file_name)

    boxes, labels, truncation = JV.read_metro_linx_label_untilt(path_to_source_labels)
    corners3d = JV.boxes_to_corners_3d(boxes).squeeze(0)
    # corners3d_xyz = corners3d[:, [0, 2, 1]]
    corners3d_xyz = corners3d
    # print(corners3d_xyz)
    # print(np.mean(corners3d, axis=0))
    # print(np.mean(corners3d_xyz, axis=0))
    # new_corners_xyz = rotate_points(corners3d_xyz)
    # new_corners = new_corners_xyz[:, [0, 2, 1]]
    new_corners = rotate_points(corners3d_xyz)
    # print(np.mean(new_corners, axis=0))
    # print(np.mean(new_corners_xyz, axis=0))
    new_box = JU.corners_to_center(new_corners).squeeze(0)
    alpha = np.arctan2(new_box[1], new_box[0])
    kitti_list = [str(labels[0]),

                  # truncated
                  str(truncation[0]),

                  # occluded
                  str(0),

                  # alpha
                  str(round(alpha, 2)),

                  # bbox
                  'Nan', 'Nan', 'Nan', 'Nan',

                  # height
                  str(round(new_box[5], 2)),
                  # width
                  str(round(new_box[4], 2)),
                  # length
                  str(round(new_box[3], 2)),

                  # x
                  str(round(new_box[0], 2)),
                  # y
                  str(round(new_box[1], 2)),
                  # z
                  str(round(new_box[2], 2)),

                  # rz (previous ry)
                  str(round(new_box[6], 2) - 1.57)]

    with open(path_to_result_labels, 'w') as out:
        out.write(' '.join(str(i) for i in kitti_list) + '\n')


if __name__ == '__main__':
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".txt"):
            rotate_txt(file_name=filename)
        else:
            continue

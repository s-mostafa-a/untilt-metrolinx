import os
import numpy as np
from jacob_land import JV, JU
from utils import rotate_points

# you may want to change these variables
INPUT_DIRECTORY = './data/labels/input'
OUTPUT_DIRECTORY = './data/labels/output'


def rotate_txt(file_name):
    path_to_source_labels = os.path.join(INPUT_DIRECTORY, file_name)
    path_to_result_labels = os.path.join(OUTPUT_DIRECTORY, file_name)

    boxes, labels, truncation = JV.read_metro_linx_label_untilt(path_to_source_labels)
    corners3d = JV.boxes_to_corners_3d(boxes)
    new_corners = rotate_points(corners3d[0])
    new_box = JU.corners_to_center(new_corners)
    alpha = np.arctan2(new_box[0, 1], new_box[0, 0])
    kitti_list = [str(labels[0]), str(truncation[0]), str(0), str(round(alpha, 2)), 'Nan', 'Nan',
                  'Nan', 'Nan', str(round(new_box[0, 5], 2)), str(round(new_box[0, 4], 2)),
                  str(round(new_box[0, 3], 2)), str(round(new_box[0, 0], 2)),
                  str(round(new_box[0, 1], 2)), str(round(new_box[0, 2], 2)),
                  str(round(new_box[0, 6], 2) - 1.57)]
    with open(path_to_result_labels, 'w') as out:
        out.write(' '.join(str(i) for i in kitti_list) + '\n')


if __name__ == '__main__':
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".txt"):
            rotate_txt(file_name=filename)
        else:
            continue

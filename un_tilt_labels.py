import os
import numpy as np
from main import read_kitti_format_for_metrolinx, box_to_corners_3d, corners_3d_to_boxes
from utils import rotate_points

# you may want to change these variables
INPUT_DIRECTORY = './data/labels/input'
OUTPUT_DIRECTORY = './data/labels/output'


def rotate_txt(file_name):
    path_to_source_labels = os.path.join(INPUT_DIRECTORY, file_name)
    path_to_result_labels = os.path.join(OUTPUT_DIRECTORY, file_name)
    kitty_obj = read_kitti_format_for_metrolinx(path_to_source_labels)[0]
    kitty_box = np.array([kitty_obj.get_3d_box_in_world_coordinates()])
    tilt_corners3d = box_to_corners_3d(kitty_box).squeeze(0)
    new_corners = rotate_points(tilt_corners3d)
    new_box = corners_3d_to_boxes(new_corners)
    alpha = np.arctan2(new_box[1], new_box[0])
    kitty_obj.set_alpha(alpha)
    kitty_obj.set_xyz(new_box[0:3])
    kitty_obj.set_hwl(new_box[3:6])
    kitty_obj.set_rotation_y(new_box[6])
    print(new_box)
    print(kitty_obj.get_line_for_txt_file())
    with open(path_to_result_labels, 'w') as out:
        out.write(kitty_obj.get_line_for_txt_file() + '\n')


if __name__ == '__main__':
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".txt"):
            rotate_txt(file_name=filename)
        else:
            continue

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
    print(new_corners)
    print(np.mean(new_corners, axis=0))
    new_box = JU.corners_to_center(new_corners).squeeze(0)
    print(new_box)
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
                  str(round(new_box[3], 2)),
                  # width
                  str(round(new_box[4], 2)),
                  # length
                  str(round(new_box[5], 2)),

                  # x
                  str(round(new_box[0], 2)),
                  # y
                  str(round(new_box[1], 2)),
                  # z
                  str(round(new_box[2], 2)),

                  # yr
                  str(round(new_box[6] - 1.57, 2))]
    # str(round(new_box[0, 5], 2)),
    # str(round(new_box[0, 4], 2)),
    # str(round(new_box[0, 3], 2)),
    # str(round(new_box[0, 0], 2)),
    # str(round(new_box[0, 1], 2)),
    # str(round(new_box[0, 2], 2)),
    # str(round(new_box[0, 6], 2) - 1.57)

if __name__ == '__main__':
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".txt"):
            rotate_txt(file_name=filename)
        else:
            continue

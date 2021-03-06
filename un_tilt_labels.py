import os
import numpy as np
from utils import rotate_points, box_to_corners_3d, extract_kitty_labels_from_file, corners_3d_to_boxes

# you may want to change these variables
INPUT_DIRECTORY = './data/labels/input'
OUTPUT_DIRECTORY = './data/labels/output'


def rotate_txt(file_name):
    path_to_source_labels = os.path.join(INPUT_DIRECTORY, file_name)
    path_to_result_labels = os.path.join(OUTPUT_DIRECTORY, file_name)

    label, truncation, occlusion, alpha, two_d_bbox, hwl, xyz, rotation_y, score = extract_kitty_labels_from_file(
        path_to_source_labels)
    box_in_world_coordinate = np.concatenate(
        (np.array([xyz[0], xyz[2], xyz[1]]), np.array([hwl[0], hwl[2], hwl[1]]), [rotation_y]))
    tilt_corners_3d = box_to_corners_3d(box_in_world_coordinate[None, ::])
    untilt_corners_3d = rotate_points(tilt_corners_3d.squeeze(0))
    xzy, hlw, rotation_z = corners_3d_to_boxes(untilt_corners_3d)
    alpha = np.arctan2(xzy[1], xzy[0])
    for_file = f'''{label} {int(truncation)} {int(occlusion)} {
    alpha} {' '.join(np.char.mod('%.2f', two_d_bbox))} {' '.join(np.char.mod('%.2f', hlw[[0, 2, 1]]))} {
    ' '.join(np.char.mod('%.2f', xzy[[0, 2, 1]]))} {round(rotation_z, 2)}\n'''

    with open(path_to_result_labels, 'w') as out:
        out.write(for_file)


if __name__ == '__main__':
    for filename in os.listdir(INPUT_DIRECTORY):
        if filename.endswith(".txt"):
            rotate_txt(file_name=filename)
        else:
            continue

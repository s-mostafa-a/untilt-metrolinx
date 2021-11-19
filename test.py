import numpy as np
from utils import rotate_points, box_to_corners_3d, extract_kitty_labels_from_file, corners_3d_to_boxes


def main():
    file_number = '1192'
    tilt_csv_file = f'./not_upload_data/point_cloud/input/xyzi_m1412_{file_number}.csv'
    tilt_label_file = f'./not_upload_data/labels/input/label_m1412_{file_number}.txt'
    label, truncation, occlusion, alpha, two_d_bbox, hwl, xyz, rotation_y, score = extract_kitty_labels_from_file(
        tilt_label_file)
    box_in_world_coordinate = np.concatenate(
        (np.array([xyz[0], xyz[2], xyz[1]]), np.array([hwl[0], hwl[2], hwl[1]]), [rotation_y]))
    tilt_corners_3d = box_to_corners_3d(box_in_world_coordinate[None, ::])
    untilt_corners_3d = rotate_points(tilt_corners_3d.squeeze(0))
    xzy, hlw, rotation_z = corners_3d_to_boxes(untilt_corners_3d)
    for_file = f'''{label} {int(truncation)} {int(occlusion)} {
    alpha} {' '.join(np.char.mod('%.2f', two_d_bbox))} {' '.join(np.char.mod('%.2f', hlw[[0, 2, 1]]))} {
    ' '.join(np.char.mod('%.2f', xzy[[0, 2, 1]]))} {round(rotation_z, 2)}'''
    print(for_file)


if __name__ == '__main__':
    main()

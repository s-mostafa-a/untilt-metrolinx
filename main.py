import numpy as np
import mayavi.mlab as mlab
from copy import deepcopy
from jacob_land.visualize_utils import read_metro_linx_label, draw_metrolinx_scene
from utils import rotate_points
import torch


# Train 0 0 0.0 Nan Nan Nan Nan 6.07 4.49 67.2 -30.0 1.69 5.97 -1.49
#                                h    w     l     x   y    z
def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa, sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2
    print()
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def center_to_eight_corners(xyz, hwl):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    height = hwl[0]
    width = hwl[1]
    length = hwl[2]
    return np.array([[x + length / 2, y, z + width / 2],
                     [x + length / 2, y, z - width / 2],
                     [x - length / 2, y, z + width / 2],
                     [x - length / 2, y, z - width / 2],
                     [x + length / 2, y - height, z + width / 2],
                     [x + length / 2, y - height, z - width / 2],
                     [x - length / 2, y - height, z + width / 2],
                     [x - length / 2, y - height, z - width / 2]])


class KittyObject(object):
    def __init__(self, line: str):
        line = line.replace('\n', '')
        line = line.split(' ')
        self._type = line[0]
        self._array = np.fromstring(', '.join(line[1:]), dtype=float, sep=',')
        self._truncation = self._array[0]
        self._occlusion = self._array[1]
        self._alpha = self._array[2]
        self._2d_bbox = self._array[3:7]
        self._hwl = self._array[7:10]
        self._xyz = self._array[10:13]
        self._rotation_y = self._array[13]
        if len(self._array) > 14:
            self._score = self._array[14]
        else:
            self._score = np.nan

    def get_xyz(self):
        return self._xyz

    def get_hwl(self):
        return self._hwl

    def get_type(self):
        return self._type

    def get_line_for_txt_file(self):
        array_str = np.char.mod('%.2f', self._array)
        return f'{self._type} {" ".join(array_str)}'


def read_kitti_format_for_metrolinx(file):
    kitty_objects = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            ko = KittyObject(line)
            kitty_objects.append(ko)
            print(ko.get_line_for_txt_file())
    return kitty_objects


def main():
    file_number = '1279'
    tilt_csv_file = f'./data/point_cloud/input/xyzi_m1412_{file_number}.csv'
    tilt_label_file = f'./data/labels/input/label_m1412_{file_number}.txt'
    tilt_pointclouds = np.genfromtxt(tilt_csv_file, delimiter=',')
    tilt_boxes, _, _ = read_metro_linx_label(tilt_label_file)
    kitty_objs = read_kitti_format_for_metrolinx(tilt_label_file)
    my_corners = center_to_eight_corners(kitty_objs[0].get_xyz(), kitty_objs[0].get_hwl())
    my_corners = my_corners[:, [0, 2, 1]]
    tilt_corners3d = boxes_to_corners_3d(tilt_boxes)
    print(my_corners)
    print(tilt_corners3d)

    # print(center_to_eight_corners())
    untilt_pointclouds = deepcopy(tilt_pointclouds)
    untilt_pointclouds[:, :3] = rotate_points(tilt_pointclouds[:, :3])
    untilt_corners3d = rotate_points(tilt_corners3d.squeeze(0))[None, ::]
    # draw_metrolinx_scene(untilt_pointclouds, untilt_corners3d)
    draw_metrolinx_scene(tilt_pointclouds, my_corners[None, ::])
    draw_metrolinx_scene(tilt_pointclouds, tilt_corners3d)
    mlab.show(stop=True)


if __name__ == '__main__':
    main()

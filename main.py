import numpy as np
import mayavi.mlab as mlab
from copy import deepcopy
from jacob_land.visualize_utils import draw_metrolinx_scene
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
    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


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

    def _hlw(self):
        return self._hwl[[0, 2, 1]]

    def _xzy(self):
        return self._xyz[[0, 2, 1]]

    def get_3d_box(self):
        return np.concatenate((self._xyz, self._hwl, [self._rotation_y]))

    def get_3d_box_in_world_coordinates(self):
        return np.concatenate((self._xzy(), self._hlw(), [self._rotation_y]))

    def get_rotation_y(self):
        return self._rotation_y


def read_kitti_format_for_metrolinx(file):
    kitty_objects = []
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            ko = KittyObject(line)
            kitty_objects.append(ko)
    return kitty_objects


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))


def corners_3d_to_boxes(corners):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    # check if 0,1,2,3 are in the same plane or 0,1,5,4 or even 0,3,7,4
    # hypothesis: 0,1,2,3 are in the same plane
    cx, cy, cz = np.mean(corners, axis=0)
    dx = distance(corners[0, :], corners[3, :])
    dy = distance(corners[0, :], corners[1, :])
    dz = distance(corners[0, :], corners[4, :])
    rz = np.arctan2(corners[3, 0] - corners[2, 0], corners[3, 1] - corners[2, 1])
    return np.array([cx, cy, cz, dx, dy, dz, -rz])


def main():
    file_number = '1192'
    tilt_csv_file = f'./not_upload_data/point_cloud/input/xyzi_m1412_{file_number}.csv'
    tilt_label_file = f'./not_upload_data/labels/input/label_m1412_{file_number}.txt'
    tilt_pointclouds = np.genfromtxt(tilt_csv_file, delimiter=',')
    kitty_objs = read_kitti_format_for_metrolinx(tilt_label_file)
    kitty_boxes = np.array([ko.get_3d_box_in_world_coordinates() for ko in kitty_objs])
    tilt_corners3d = boxes_to_corners_3d(kitty_boxes)
    kitty_boxes[0, 6] = 0
    tilt_corners3d_1 = boxes_to_corners_3d(kitty_boxes)
    corners_3d_to_boxes(tilt_corners3d[0])
    print("that", kitty_objs[0].get_rotation_y())

    exit(0)
    # print("mean", np.mean(tilt_corners3d, axis=1))
    #
    # print("center", kitty_objs[0]._xzy())
    # print("hlw", kitty_objs[0]._hlw())
    #
    # print("------------------")
    # print("mine", corners_3d_to_boxes(tilt_corners3d[0]))
    # print("that", kitty_objs[0].get_rotation_y())
    # exit(0)
    # untilt_pointclouds = deepcopy(tilt_pointclouds)
    # untilt_pointclouds[:, :3] = rotate_points(tilt_pointclouds[:, :3])
    # untilt_corners3d = rotate_points(tilt_corners3d.squeeze(0))[None, ::]
    # draw_metrolinx_scene(untilt_pointclouds, untilt_corners3d)
    # draw_metrolinx_scene(tilt_pointclouds, my_corners[None, ::])
    draw_metrolinx_scene(tilt_pointclouds, tilt_corners3d)
    draw_metrolinx_scene(tilt_pointclouds, tilt_corners3d_1)
    mlab.show(stop=True)


if __name__ == '__main__':
    main()

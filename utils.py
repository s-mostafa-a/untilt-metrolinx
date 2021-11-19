import numpy as np
import torch

ROTATION_MATRIX = np.array([[0.99648296, 0.03442719, -0.07639682],
                            [-0.01334566, 0.9652704, 0.26091177],
                            [0.08272604, -0.25897457, 0.96233496]])


def rotate_points(points):
    rotated_points_transpose = ROTATION_MATRIX @ points.T
    rotated_points_transpose = rotated_points_transpose * np.array([[-1], [-1], [1]])
    rotated_points = rotated_points_transpose.T
    return rotated_points


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


def box_to_corners_3d(boxes3d):
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


def extract_kitty_labels_from_file(file):
    with open(file) as f:
        lines = f.readlines()
        line = lines[0]
    line = line.replace('\n', '')
    line = line.split(' ')
    label = line[0]
    array = np.fromstring(', '.join(line[1:]), dtype=float, sep=',')
    truncation = array[0]
    occlusion = array[1]
    alpha = array[2]
    two_d_bbox = array[3:7]
    hwl = array[7:10]
    xyz = array[10:13]
    rotation_y = array[13]
    if len(array) > 14:
        score = array[14]
    else:
        score = np.nan
    return label, truncation, occlusion, alpha, tuple(two_d_bbox), tuple(hwl), tuple(xyz), rotation_y, score


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
    return np.array([cx, cy, cz]), np.array([dx, dy, dz]), -rz

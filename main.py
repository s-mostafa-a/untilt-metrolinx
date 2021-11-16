import numpy as np
import mayavi.mlab as mlab
from copy import deepcopy
from jacob_land.visualize_utils import read_metro_linx_label, boxes_to_corners_3d, draw_metrolinx_scene
from utils import rotate_points

file_number = '1279'
tilt_csv_file = f'./data/point_cloud/input/xyzi_m1412_{file_number}.csv'
tilt_label_file = f'./data/labels/input/label_m1412_{file_number}.txt'
tilt_pointclouds = np.genfromtxt(tilt_csv_file, delimiter=',')
tilt_boxes, _, _ = read_metro_linx_label(tilt_label_file)
tilt_corners3d = boxes_to_corners_3d(tilt_boxes)

untilt_pointclouds = deepcopy(tilt_pointclouds)
untilt_pointclouds[:, :3] = rotate_points(tilt_pointclouds[:, :3])
untilt_corners3d = rotate_points(tilt_corners3d.squeeze(0))[None, ::]
draw_metrolinx_scene(untilt_pointclouds, untilt_corners3d)
draw_metrolinx_scene(tilt_pointclouds, tilt_corners3d)
mlab.show(stop=True)

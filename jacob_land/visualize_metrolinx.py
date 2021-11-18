#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:17:20 2021

@author: jacob
visualization for metrolinx dataset
"""

import numpy as np
import mayavi.mlab as mlab

from visualize_utils import read_metro_linx_label, boxes_to_corners_3d, draw_metrolinx_scene


# file =

# for file in files:
# place your point cloud file path here
file = '1279'
tilt_csv_file = f'../data/point_cloud/input/xyzi_m1412_{file}.csv'
untilt_csv_file = f'../data/point_cloud/output/xyzi_m1412_{file}.csv'

# place your label file here
tilt_label_file = f'../data/labels/input/label_m1412_{file}.txt'
untilt_label_file = f'../data/labels/output/label_m1412_{file}.txt'

# loads point cloud
tilt_pointclouds = np.genfromtxt(tilt_csv_file, delimiter=',')
untilt_pointclouds = np.genfromtxt(untilt_csv_file, delimiter=',')

# loads labels
tilt_boxes, _, _ = read_metro_linx_label(tilt_label_file)
untilt_boxes, _, _ = read_metro_linx_label(untilt_label_file)

# converted kitti object into 8 point corners
tilt_corners3d = boxes_to_corners_3d(tilt_boxes)
untilt_corners3d = boxes_to_corners_3d(untilt_boxes)

# draw_scene
draw_metrolinx_scene(tilt_pointclouds, tilt_corners3d)
draw_metrolinx_scene(untilt_pointclouds, untilt_corners3d)
# mlab.savefig(filename='test.png')
mlab.show(stop=True)

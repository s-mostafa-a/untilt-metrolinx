#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:17:20 2021

@author: jacob
visualization for metrolinx dataset
"""
import mayavi.mlab as mlab

import numpy as np

from jacob_land import visualize_utils as V

# file =

# for file in files:
# place your point cloud file path here
csv_file = '../data/point_cloud/input/xyzi_sample1.csv'

# place your label file here
label_file = '../data/labels/output/label_m1412_1279.txt'

# loads point cloud
pointclouds = np.genfromtxt(csv_file, delimiter=',')

# loads labels
boxes, _, _ = V.read_metro_linx_label(label_file)

# converted kitti object into 8 point corners
corners3d = V.boxes_to_corners_3d(boxes)

# draw_scene
V.draw_metrolinx_scene(pointclouds, corners3d)
# mlab.savefig(filename='test.png')
mlab.show(stop=True)

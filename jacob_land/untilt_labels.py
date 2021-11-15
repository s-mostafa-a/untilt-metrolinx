#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:41:04 2021

@author: jacob
"""
import numpy as np

from .utils import untilt_ptc, corners_to_center
from .visualize_utils import read_metro_linx_label_untilt, boxes_to_corners_3d

# place your point cloud file path here
csv_file = '../not_upload_data/1187-1300/point_cloud/xyzi_m1412_1287.csv'

# place your label file here
label_file = '../not_upload_data/1187-1300/labels/label_m1412_1287.txt'

# loads point cloud
pointclouds = np.genfromtxt(csv_file, delimiter=',')

# loads rotational matrix
rot = np.array([[0.99648296, 0.03442719, -0.07639682],
                [-0.01334566, 0.9652704, 0.26091177],
                [0.08272604, -0.25897457, 0.96233496]])

new_label_file = '../label_m1412_1279.txt'

new_csv_file = '../xyzi_m1412_1279.csv'
# loads labels
boxes, labels, truncation = read_metro_linx_label_untilt(label_file)
corners3d = boxes_to_corners_3d(boxes)

# xyz = boxes[:,:3]

# new_xyz = untilt_ptc(xyz, rot)
new_corners = []
for i in range(boxes.shape[0]):
    new_corners.append(untilt_ptc(corners3d[i], rot))
new_pointclouds = untilt_ptc(pointclouds, rot)
# converted kitti object into 8 point corners

np.savetxt(new_csv_file, new_pointclouds, delimiter=",")
label_list = []
for i in range(len(new_corners)):
    new_box = corners_to_center(new_corners[i])
    alpha = np.arctan2(new_box[0, 1], new_box[0, 0])
    kitti_list = [str(labels[i]), str(truncation[i]), str(0), str(round(alpha, 2)), 'Nan', 'Nan',
                  'Nan', 'Nan', str(round(new_box[0, 5], 2)), str(round(new_box[0, 4], 2)),
                  str(round(new_box[0, 3], 2)), str(round(new_box[0, 0], 2)),
                  str(round(new_box[0, 1], 2)), str(round(new_box[0, 2], 2)),
                  str(round(new_box[0, 6], 2) - 1.57)]
    label_list.append(kitti_list)

with open(new_label_file, 'a') as out:
    for kitti_label in label_list:
        out.write(' '.join(str(i) for i in kitti_label) + '\n')

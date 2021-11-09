import numpy as np
import mayavi.mlab as mlab
import open3d as o3d
import torch
from jacob_files import visualize_utils as vis

# file =

# for file in files:
# place your point cloud file path here
csv_file = '/home/jacob/metrolinx_dataset/1412_1419/untilted_outser/xyzi_m1412_1279.csv'

# place your label file here
label_file = '/home/jacob/metrolinx_dataset/1412_1419/test/label_m1412_1279.txt'

# loads point cloud
pointclouds = np.genfromtxt(csv_file, delimiter=',')

# loads labels
boxes, _, _ = vis.read_metro_linx_label(label_file)

# converted kitti object into 8 point corners
corners3d = vis.boxes_to_corners_3d(boxes)

# draw_scene
vis.draw_metrolinx_scene(pointclouds, corners3d)
# mlab.savefig(filename='test.png')
mlab.show(stop=True)

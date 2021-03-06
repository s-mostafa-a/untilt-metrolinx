#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 17:17:20 2021

@author: jacob
visualization for metrolinx dataset
"""

import mayavi.mlab as mlab
import numpy as np
import torch
import laspy

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def read_las_file(data_filename):
    '''
    loading las file
    ---------------------------------------
    input: las file name
    output: np array with dimension of [num of points, labels]
            labels are [x,y,z, intensity]
    '''
    inFile = laspy.file.File(data_filename)
    data = np.zeros([len(inFile), 3], dtype=None)
    data[:, 0] = inFile.X * inFile.header.scale[0]  # + inFile.header.offset[0]
    data[:, 1] = inFile.Y * inFile.header.scale[1]  # + inFile.header.offset[1]
    data[:, 2] = inFile.Z * inFile.header.scale[2]  # + inFile.header.offset[2]

    return data


def roty(t):
    """
    Rotation about the y-axis.
    """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


class Box3D(object):
    """
    Represent a 3D box corresponding to data in label.txt
    """

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        # data[1:] = [float(x) for x in data[1:]]

        self.type = data[0]
        self.truncation = float(data[1])
        self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = float(data[3])  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        if data[4] != 'None':
            self.xmin = float(data[4])  # left
            self.ymin = float(data[5])  # top
            self.xmax = float(data[6])  # right
            self.ymax = float(data[7])  # bottom
            self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        else:
            self.xmin = data[4]  # left
            self.ymin = data[5]  # top
            self.xmax = data[6]  # right
            self.ymax = data[7]  # bottom
            self.box2d = None

        # extract 3d bounding box information
        self.h = float(data[8])  # box height
        self.w = float(data[9])  # box width
        self.l = float(data[10])  # box length (in meters)
        self.t = (
            float(data[11]), float(data[12]), float(data[13]))  # location (x,y,z) in camera coord.
        self.ry = float(data[14])  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
        # self.score = float(data[15])

    def in_camera_coordinate(self, is_homogenous=False):
        # 3d bounding box dimensions
        l = self.l
        w = self.w
        h = self.h

        # 3D bounding box vertices [3, 8]
        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        box_coord = np.vstack([x, y, z])

        # Rotation
        R = roty(self.ry)  # [3, 3]
        points_3d = R @ box_coord

        # Translation
        points_3d[0, :] = points_3d[0, :] + self.t[0]
        points_3d[1, :] = points_3d[1, :] + self.t[1]
        points_3d[2, :] = points_3d[2, :] + self.t[2]

        if is_homogenous:
            points_3d = np.vstack((points_3d, np.ones(points_3d.shape[1])))

        return points_3d


def read_linx_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    # mask_scores = pred_scores > 0.80
    boxes = []
    labels = []
    # 3d_boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    for i in range(len(objects)):
        an_object = objects[i]
        box = [an_object.t[0], an_object.t[1], an_object.t[2], an_object.l, an_object.w,
               an_object.h,
               0]  # this might be revisited, i think heading is worngly calculated....
        boxes.append(box)
        labels.append(an_object.type)
    boxes = np.array(boxes)
    # labels = np.array(labels)#need to convert labels into numbers
    labels = np.ones((len(objects),), dtype=int)
    mask = np.ones((len(objects), 1),
                   dtype=int)  # supposedly there should be score to threshold based on score, but currently, score is not given.
    score = np.ones((len(objects),))  # scores are not given, so currently just putting 1
    return mask, boxes, labels, score


def read_kitti_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    # mask_scores = pred_scores > 0.80
    boxes = []
    labels = []
    # 3d_boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    for i in range(len(objects)):
        an_object = objects[i]
        box = [an_object.t[2], -an_object.t[0], -an_object.t[1], an_object.w, an_object.l,
               an_object.h,
               an_object.ry]  # this might be revisited, i think heading is worngly calculated....
        boxes.append(box)
        labels.append(an_object.type)
    boxes = np.array(boxes)
    labels = np.array(labels)  # need to convert labels into numbers
    # out_labels = np.ones((len(objects), ), dtype=int)
    # mask = np.ones((len(objects),1), dtype=int)#supposedly there should be score to threshold based on score, but currently, score is not given.
    return boxes, labels


def read_metro_linx_label_preprocess(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    # mask_scores = pred_scores > 0.80
    boxes = []
    labels = []
    t = []
    # 3d_boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    for i in range(len(objects)):
        an_object = objects[i]
        box = [an_object.t[0], an_object.t[2], an_object.t[1], an_object.w, an_object.l,
               an_object.h, np.radians(
                an_object.ry)]  # this might be revisited, i think heading is worngly calculated....
        boxes.append(box)
        labels.append(an_object.type)
        t.append(an_object.truncation)
    boxes = np.array(boxes)
    labels = np.array(labels)  # need to convert labels into numbers
    # out_labels = np.ones((len(objects), ), dtype=int)
    # mask = np.ones((len(objects),1), dtype=int)#supposedly there should be score to threshold based on score, but currently, score is not given.
    return boxes, labels, t


def read_metro_linx_label_untilt(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    # mask_scores = pred_scores > 0.80
    boxes = []
    labels = []
    t = []
    # 3d_boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    for i in range(len(objects)):
        an_object = objects[i]
        box = [an_object.t[0], an_object.t[2], an_object.t[1], an_object.w, an_object.l,
               an_object.h,
               an_object.ry]  # this might be revisited, i think heading is worngly calculated....
        boxes.append(box)
        labels.append(an_object.type)
        t.append(an_object.truncation)
    boxes = np.array(boxes)
    labels = np.array(labels)  # need to convert labels into numbers
    # out_labels = np.ones((len(objects), ), dtype=int)
    # mask = np.ones((len(objects),1), dtype=int)#supposedly there should be score to threshold based on score, but currently, score is not given.
    return boxes, labels, t


def read_metro_linx_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    # load as list of Object3D
    objects = [Box3D(line) for line in lines]
    # mask_scores = pred_scores > 0.80
    boxes = []
    labels = []
    t = []
    # 3d_boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    for i in range(len(objects)):
        an_object = objects[i]
        box = [an_object.t[0], an_object.t[2], an_object.t[1], an_object.w, an_object.l,
               an_object.h,
               an_object.ry]  # this might be revisited, i think heading is worngly calculated....
        boxes.append(box)
        labels.append(an_object.type)
        t.append(an_object.truncation)
    boxes = np.array(boxes)
    labels = np.array(labels)  # need to convert labels into numbers
    # out_labels = np.ones((len(objects), ), dtype=int)
    # mask = np.ones((len(objects),1), dtype=int)#supposedly there should be score to threshold based on score, but currently, score is not given.
    return boxes, labels, t


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


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True, title=None):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=title, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()),
                          mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None,
                figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None,
                figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None,
                figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1,
                figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1,
                figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1,
                figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1,
                figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, mask, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        # ref_corners3d = ref_boxes
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores,
                                 max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                # mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d, fig=fig, color=cur_color, cls=ref_scores[mask],
                                     max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def fix_bb(points, corners3d):
    # fig = visualize_pts(points)
    # fig = draw_multi_grid_range(fig, bv_range=(0, -80, 80, 80))
    max_point = np.max(points[:, 0])
    max_mask = corners3d[:, :, 0] > max_point
    corners3d[:, :, 0][max_mask] = max_point

    min_point = np.min(points[:, 0])
    min_mask = corners3d[:, 0] < min_point
    min_mask = corners3d[:, :, 0] < min_point
    corners3d[:, :, 0][min_mask] = min_point
    # fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)
    # mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return corners3d


def draw_metrolinx_scene(points, corners3d, title=None):
    fig = visualize_pts(points, title=title)
    fig = draw_multi_grid_range(fig, bv_range=(0, -80, 80, 80))
    fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return corners3d


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500,
                   tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3),
                            color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3),
                            color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                        tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                    tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color,
                    tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig


def main():
    file = '1279'
    tilt_csv_file = f'./data/point_cloud/input/xyzi_m1412_{file}.csv'
    untilt_csv_file = f'./data/point_cloud/output/xyzi_m1412_{file}.csv'

    tilt_label_file = f'./data/labels/input/label_m1412_{file}.txt'
    untilt_label_file = f'./data/labels/output/label_m1412_{file}.txt'

    tilt_pointclouds = np.genfromtxt(tilt_csv_file, delimiter=',')
    untilt_pointclouds = np.genfromtxt(untilt_csv_file, delimiter=',')

    tilt_boxes, _, _ = read_metro_linx_label(tilt_label_file)
    untilt_boxes, _, _ = read_metro_linx_label(untilt_label_file)

    tilt_corners3d = boxes_to_corners_3d(tilt_boxes)
    untilt_corners3d = boxes_to_corners_3d(untilt_boxes)

    draw_metrolinx_scene(tilt_pointclouds, tilt_corners3d)
    draw_metrolinx_scene(untilt_pointclouds, untilt_corners3d)
    mlab.show(stop=True)


if __name__ == '__main__':
    main()

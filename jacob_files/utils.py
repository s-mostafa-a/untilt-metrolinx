#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:48:27 2021

@author: jacob
"""
import numpy as np
from scipy.linalg import svd

def translation_and_roation(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) @ BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       #print "Reflection detected"
       Vt[2,:] @= -1
       R = Vt.T * U.T

    t = -R@centroid_A.T + centroid_B.T

    #print t

    return R, t

def distances(a,b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)

def corners_to_center(corners_3d):
    rz = np.arctan2(corners_3d[3,1] - corners_3d[2,1], corners_3d[3,0] - corners_3d[2,0])
    cx = (corners_3d[2,0] + corners_3d[4,0])/2
    cy = (corners_3d[2,1] + corners_3d[4,1])/2
    cz = (corners_3d[2,2] + corners_3d[4,2])/2
    
    dx = distances(corners_3d[2], corners_3d[3])
    dy = distances(corners_3d[2], corners_3d[1])
    dz = distances(corners_3d[2], corners_3d[6])
    
    return np.array([[cx, cy, cz, dx, dy, dz, rz]])

def untilt_ptc(pointclouds, rot):
    mat_pnts_in_temp = pointclouds[:, 0:3]
    mat_pnts_in_temp = mat_pnts_in_temp.T
    
    mat_pnts_out_temp = np.matmul(rot, mat_pnts_in_temp)
    mat_pnts_out_temp = mat_pnts_out_temp.T
    
    mat_pnts_out_temp[:,0] = -1.0*mat_pnts_out_temp[:,0]
    mat_pnts_out_temp[:,1] = -1.0*mat_pnts_out_temp[:,1]
    
    mat_pnts_out = pointclouds
    mat_pnts_out[:, 0:3] = mat_pnts_out_temp
    return mat_pnts_out
     
      
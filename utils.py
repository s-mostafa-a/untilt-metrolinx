import numpy as np

ROTATION_MATRIX = np.array([[0.99648296, 0.03442719, -0.07639682],
                            [-0.01334566, 0.9652704, 0.26091177],
                            [0.08272604, -0.25897457, 0.96233496]])


def rotate_points(points):
    rotated_points_transpose = ROTATION_MATRIX @ points.T
    rotated_points_transpose = rotated_points_transpose * np.array([[-1], [-1], [1]])
    rotated_points = rotated_points_transpose.T
    return rotated_points

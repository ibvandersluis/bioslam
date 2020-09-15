import math
import numpy as np

def closest(cones):
    """
    Returns the distances of the closest right and left cones, respectively
    
    :param cones: An array of x-y cone positions, relative to vehicle
    :return:
        - right: The distance to the closest cone on the right
        - left: The distance to the closest cone on the left
    """
    right_cones = cones
    left_cones = np.zeros(0, len(cones[0]))
    right = None
    left = None

    for i in range(len(cones)):
        if (cones[i, 0] < 0):
            left_cones = np.vstack((left_cones, cones[i]))
            right_cones = np.delete(right_cones, i, 0)

    right = math.hypot(right_cones[0, 0], right_cones[0, 1])
    for j in range(len(right_cones)):
        d = math.hypot(right_cones[j, 0], right_cones[j, 1])
        if (d < right):
            right = d

    left = math.hypot(left_cones[0, 0], left_cones[0, 1])
    for k in range(len(left_cones)):
        d = math.hypot(left_cones[k, 0], left_cones[k, 1])
        if (d < left):
            left = d

    return right, left
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


# Convert angles into x-y coordinates
def angles2xy(th1, th2, l1 = 1, l2 = 1):
    x1 = l1 * np.sin(th1)
    y1 = -l1 * np.cos(th1)
    x2 = x1 + (l2 * np.sin(th2))
    y2 = y1 - (l2 * np.cos(th2))
    return x1, y1, x2, y2

# Ensure that the angles stays between [-pi, pi)
def wrap_angles(angle, wrap=True):
    # result = ( angle ) % (2 * np.pi ) # [0, 2pi)
    if wrap:
        result = ( angle + np.pi) % (2 * np.pi ) - np.pi   # [-pi, pi)
    else:
        result = angle

    return result
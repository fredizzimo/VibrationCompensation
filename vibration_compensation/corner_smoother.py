# This is mostly based on the following paper:
# A Real-Time C3 Continuous Local Corner Smoothing and Interpolation Algorithm for CNC Machine Tools

import numpy as np
np.set_printoptions(suppress=True)

class CornerSmoother(object):
    def __init__(self, maximum_error):
        self.maximum_error = maximum_error

    def generate_corners(self, data):
        P0 = np.column_stack((data["start_x"][:-1], data["start_y"][:-1]))
        P1 = np.column_stack((data["x"][:-1], data["y"][:-1]))
        P2 = np.column_stack((data["x"][1:], data["y"][1:]))
        vector_a = P0 - P1
        vector_b = P2 - P1
        length_a = np.linalg.norm(vector_a, axis=1)
        length_b = np.linalg.norm(vector_b, axis=1)
        with np.errstate(invalid='ignore'):
            dot_product = np.einsum('ij,ij->i', vector_a, vector_b)
            angle = np.arccos(dot_product / (length_a * length_b))

        for i in range(1000):
            print(P0[i], P1[i], P2[i], length_b[i], length_b[i], np.degrees(angle[i]))

        print(P0.shape)



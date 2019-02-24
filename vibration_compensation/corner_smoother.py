# This is mostly based on the following paper:
# A Real-Time C3 Continuous Local Corner Smoothing and Interpolation Algorithm for CNC Machine Tools

import numpy as np
np.set_printoptions(suppress=True)
#from vibration_compensation import PHSpline
from .phspline import PHSpline
from .data import Data

class CornerSmoother(object):
    def __init__(self, maximum_error):
        self.maximum_error = maximum_error

    def generate_corners(self, data: Data):
        P0 = data.start_xy[:-1]
        P1 = data.end_xy[:-1]
        P2 = data.end_xy[1:]
        # Some lines have zero length, but let's still do the calculations and ignore them later
        with np.errstate(invalid="ignore"):
            vector_a = P1 - P0
            vector_b = P2 - P0
            length_a = np.linalg.norm(vector_a, axis=1)
            length_b = np.linalg.norm(vector_b, axis=1)
            T0 = vector_a / length_a[:, np.newaxis]
            T1 = vector_b / length_b[:, np.newaxis]
            T0_plus_T1 = T0 + T1
            T2 = T0_plus_T1 / np.linalg.norm(T0_plus_T1, axis=1)[:, np.newaxis]
            dot_product = np.einsum('ij,ij->i', T0, T1)
            angle = np.arccos(dot_product)
            valid_segments = np.greater(angle, 0)

        # Eq 19 helper variables
        length_a = length_a[valid_segments]
        length_b = length_b[valid_segments]
        T0 = T0[valid_segments]
        T1 = T1[valid_segments]
        T2 = T2[valid_segments]
        angle = angle[valid_segments]
        half_angle = angle / 2.0
        cos_half_angle = np.cos(half_angle)
        P1 = P1[valid_segments]

        def eq_l(): #eq 23
            a = 181.0 + 50.0 / cos_half_angle
            min_1 = 21.0 * length_a / a
            min_2 = 21.0 * length_b / a

            b = 355.0 / 672.0
            c = 21.0 * cos_half_angle
            d = 25.0 / c
            e = b + d
            f = e * np.sin(half_angle)
            min_3 = self.maximum_error / f
            return np.amin((min_1, min_2, min_3), axis=0)

        def eq_l_prime():
            a = 55.0 / 42.0
            b = 21.0 * cos_half_angle
            c = 25.0 / b
            return a + c


        l = eq_l()
        l_prime = eq_l_prime()

        B0 = P1 - (3.0*l + l_prime)[:, np.newaxis] * T0
        B1 = P1 - (2.0*l + l_prime)[:, np.newaxis] * T0
        B2 = P1 - (l + l_prime)[:, np.newaxis] * T0
        B3 = P1 - (l_prime)[:, np.newaxis] * T0
        B4 = B3 + ((5.0 / 6.0) * l)[:, np.newaxis] * T0 + ((1.0 / 6.0)*l)[:, np.newaxis] * T2
        B5 = B4 + ((10.0 / 21.0) * l)[:, np.newaxis] * T0 + ((11.0 / 21.0)*l)[:, np.newaxis] * T2
        B6 = B5 + l[:, np.newaxis] * T2
        B7 = B6 + ((10.0 / 21.0) * l)[:, np.newaxis] * T1 + ((11.0 / 21.0)*l)[:, np.newaxis] * T2
        B8 = P1 - (l_prime)[:, np.newaxis] * T1
        B9 = P1 - (l + l_prime)[:, np.newaxis] * T1
        B10 = P1 - (2.0*l + l_prime)[:, np.newaxis] * T1
        B11 = P1 - (3.0*l + l_prime)[:, np.newaxis] * T1

        valid_segments_mapper = np.empty(data.curve.shape[0], dtype=valid_segments.dtype)
        valid_segments_mapper[:-1] = valid_segments
        valid_segments_mapper[-1] = False
        data.curve[valid_segments_mapper,0] = B0
        data.curve[valid_segments_mapper,1] = B1
        data.curve[valid_segments_mapper,2] = B2
        data.curve[valid_segments_mapper,3] = B3
        data.curve[valid_segments_mapper,4] = B4
        data.curve[valid_segments_mapper,5] = B5
        data.curve[valid_segments_mapper,6] = B6
        data.curve[valid_segments_mapper,7] = B7
        data.curve[valid_segments_mapper,8] = B8
        data.curve[valid_segments_mapper,9] = B9
        data.curve[valid_segments_mapper,10] = B10
        data.curve[valid_segments_mapper,11] = B11

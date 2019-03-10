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
        # Some lines have zero length, but let's still do the calculations and ignore them later
        with np.errstate(invalid="ignore"):
            vectors = data.end_xy.T - data.start_xy.T
            lengths = np.linalg.norm(vectors, axis=0)
            normalized_vecs = vectors / lengths
            length_a = lengths[:-1]
            length_b = lengths[1:]
            T0 = normalized_vecs[:,:-1]
            T1 = normalized_vecs[:,1:]
            T0_plus_T1 = T0 + T1
            T2 = T0_plus_T1 / np.linalg.norm(T0_plus_T1, axis=0)
            dot_product = np.einsum('ji,ji->i', T0, T1)
            angle = np.arccos(dot_product)
            valid_segments = np.greater(angle, 1e-6)

        # Eq 19 helper variables
        length_a = length_a[valid_segments]
        length_b = length_b[valid_segments]
        T0 = T0[:,valid_segments]
        T1 = T1[:,valid_segments]
        T2 = T2[:,valid_segments]
        P1 = data.end_xy[:-1].T[:,valid_segments]
        angle = angle[valid_segments]
        half_angle = angle / 2.0
        cos_half_angle = np.cos(half_angle)

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
            d = a + c
            return d * l


        l = eq_l()
        l_prime = eq_l_prime()

        B0 = P1 - (3.0*l + l_prime) * T0
        B1 = P1 - (2.0*l + l_prime) * T0
        B2 = P1 - (l + l_prime) * T0
        B3 = P1 - l_prime * T0
        B4 = B3 + ((5.0 / 6.0) * l) * T0 + ((1.0 / 6.0) * l) * T2
        B5 = B4 + ((10.0 / 21.0) * l) * T0 + ((11.0 / 21.0) * l) * T2
        B6 = B5 + l * T2
        B7 = B6+ ((10.0 / 21.0) * l) * T1 + ((11.0 / 21.0) * l) * T2
        B8 = P1 + l_prime * T1
        B9 = P1 + (l + l_prime) * T1
        B10 = P1 + (2.0 * l + l_prime) * T1
        B11 = P1 + (3.0 * l + l_prime) * T1

        curves = np.empty((data.start_xy.shape[0], 3, 12, 2))
        curves[:,:,0,0] = np.nan

        end_segment_mapper = np.empty(data.curve.shape[0], dtype=valid_segments.dtype)
        end_segment_mapper[:-1] = valid_segments
        end_segment_mapper[-1] = False
        curves[end_segment_mapper,2,0] = B0.T
        curves[end_segment_mapper,2,1] = B1.T
        curves[end_segment_mapper,2,2] = B2.T
        curves[end_segment_mapper,2,3] = B3.T
        curves[end_segment_mapper,2,4] = B4.T
        curves[end_segment_mapper,2,5] = B5.T
        curves[end_segment_mapper,2,6] = B6.T
        curves[end_segment_mapper,2,7] = B7.T
        curves[end_segment_mapper,2,8] = B8.T
        curves[end_segment_mapper,2,9] = B9.T
        curves[end_segment_mapper,2,10] = B10.T
        curves[end_segment_mapper,2,11] = B11.T
        start_segment_mapper = np.empty(data.curve.shape[0], dtype=valid_segments.dtype)
        start_segment_mapper[1:] = valid_segments
        start_segment_mapper[0] = False
        curves[start_segment_mapper,0,0] = B0.T
        curves[start_segment_mapper,0,1] = B1.T
        curves[start_segment_mapper,0,2] = B2.T
        curves[start_segment_mapper,0,3] = B3.T
        curves[start_segment_mapper,0,4] = B4.T
        curves[start_segment_mapper,0,5] = B5.T
        curves[start_segment_mapper,0,6] = B6.T
        curves[start_segment_mapper,0,7] = B7.T
        curves[start_segment_mapper,0,8] = B8.T
        curves[start_segment_mapper,0,9] = B9.T
        curves[start_segment_mapper,0,10] = B10.T
        curves[start_segment_mapper,0,11] = B11.T

        middle_start = np.array(data.start_xy.T, copy=True)
        middle_end = np.array(data.end_xy.T, copy=True)

        middle_start[:,start_segment_mapper] = B11
        middle_end[:,end_segment_mapper] = B0

        a = (middle_end - middle_start).T / 11.0

        curves[:,1,0] = middle_start.T
        curves[:,1,1] = curves[:,1,0] + a
        curves[:,1,2] = curves[:,1,1] + a
        curves[:,1,3] = curves[:,1,2] + a
        curves[:,1,4] = curves[:,1,3] + a
        curves[:,1,5] = curves[:,1,4] + a
        curves[:,1,6] = curves[:,1,5] + a
        curves[:,1,7] = curves[:,1,6] + a
        curves[:,1,8] = curves[:,1,7] + a
        curves[:,1,9] = curves[:,1,8] + a
        curves[:,1,10] = curves[:,1,9] + a
        curves[:,1,11] = curves[:,1,10] + a

        curves2 = curves.reshape((3*curves.shape[0], 12, 2))
        valid_curves = ~np.isnan(curves2[:,0,0])
        data.curves = np.array(curves2[valid_curves], copy=True)

        data.curve[end_segment_mapper,0] = B0.T
        data.curve[end_segment_mapper,1] = B1.T
        data.curve[end_segment_mapper,2] = B2.T
        data.curve[end_segment_mapper,3] = B3.T
        data.curve[end_segment_mapper,4] = B4.T
        data.curve[end_segment_mapper,5] = B5.T
        data.curve[end_segment_mapper,6] = B6.T
        data.curve[end_segment_mapper,7] = B7.T
        data.curve[end_segment_mapper,8] = B8.T
        data.curve[end_segment_mapper,9] = B9.T
        data.curve[end_segment_mapper,10] = B10.T
        data.curve[end_segment_mapper,11] = B11.T

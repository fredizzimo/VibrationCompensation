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

        T0_l = T0 * l
        T0_l_prime = T0 * l_prime
        T1_l = T1 * l
        T1_l_prime = T1 * l_prime
        T2_l = T2 * l

        C0 = P1 - 3.0*T0_l - T0_l_prime
        C1 = P1 - (5.0/2.0)*T0_l - T0_l_prime
        C2 = P1 - 2.0*T0_l - T0_l_prime
        C3 = P1 - (3.0/2.0)*T0_l - T0_l_prime
        C4 = P1 - (97.0/96.0)*T0_l - T0_l_prime + T2_l/96.0
        C5 = P1 - (23.0/42.0)*T0_l - T0_l_prime + T2_l/21.0
        C6 = P1 - (25.0/192.0)*T0_l - T0_l_prime + (25.0 / 192.0)*T2_l
        C7 = P1 + (51.0/224.0)*T0_l - T0_l_prime + (5.0/1344.0)*T1_l + (361.0/1344.0)*T2_l
        C8 = (P1 + (173.0/336.0)*T0_l - (255.0/256.0)*T0_l_prime +
             (5.0/336.0)*T1_l + T1_l_prime/256.0 +
             (2423.0/5376.0)*T2_l)
        C9 = (P1 + (2591.0/3584.0)*T0_l - (251.0/256.0)*T0_l_prime + (127.0/3584.0)*T1_l +
              (5.0/256.0)*T1_l_prime + (577.0/896.0)*T2_l)
        C10 = (P1 + (1521.0/1792.0)*T0_l - (121.0/128.0)*T0_l_prime + (121.0/1792.0)*T1_l +
               (7.0/128.0)*T1_l_prime + (363.0/448.0)*T2_l)
        C11 = (P1 + (3215.0/3584.0)*T0_l - (227.0/256.0)*T0_l_prime + (415.0/3584.0)*T1_l +
               (29.0/256.0)*T1_l_prime + (825.0/896.0)*T2_l)

        D0 = C11
        D1 = (P1 + (121.0/128.0)*T0_l - (53.0/64.0)*T0_l_prime + (21.0/128.0)*T1_l +
              (11.0/64.0)*T1_l_prime + (33.0/32.0)*T2_l)
        D2 = (P1 + (469.0/512.0)*T0_l - (191.0/256.0)*T0_l_prime + (117.0/512.0)*T1_l +
              (65.0/256.0)*T1_l_prime + (139.0/128.0)*T2_l)
        D3 = (P1 + (625.0/768.0)*T0_l - (163.0/256.0)*T0_l_prime + (241.0/768.0)*T1_l +
              (93.0/256.0)*T1_l_prime + (811.0/768.0)*T2_l)
        D4 = (P1 + (125.0/192.0)*T0_l - T0_l_prime/2.0 + (41.0/96.0)*T1_l + T1_l_prime/2.0 +
              (59.0/64.0)*T2_l)
        D5 = (P1 + (605.0/1344.0)*T0_l - (11.0/32.0)*T0_l_prime + (65.0/112.0)*T1_l +
              (21.0/32.0)*T1_l_prime + (925.0/1344.0)*T2_l)
        D6 = (P1 + (55.0/224.0)*T0_l - (3.0/16.0)*T0_l_prime + (533.0/672.0)*T1_l +
              (13.0/16.0)*T1_l_prime + (67.0/168.0)*T2_l)
        D7 = (P1 + (55/672.0)*T0_l - T0_l_prime/16.0 + (367.0/336.0)*T1_l +
              (15.0/16.0)*T1_l_prime + (31.0/224.0)*T2_l)
        D8 = P1 + (3.0/2.0)*T1_l + T1_l_prime
        D9 = P1 + 2.0*T1_l + T1_l_prime
        D10 = P1 + (5.0/2.0)*T1_l + T1_l_prime
        D11 = P1 + 3.0*T1*l + T1_l_prime

        curves = np.empty((data.start_xy.shape[0], 3, 12, 2))
        curves[:,:,0,0] = np.nan

        end_segment_mapper = np.empty(data.start_xy.shape[0], dtype=valid_segments.dtype)
        end_segment_mapper[:-1] = valid_segments
        end_segment_mapper[-1] = False
        curves[end_segment_mapper,2,0] = C0.T
        curves[end_segment_mapper,2,1] = C1.T
        curves[end_segment_mapper,2,2] = C2.T
        curves[end_segment_mapper,2,3] = C3.T
        curves[end_segment_mapper,2,4] = C4.T
        curves[end_segment_mapper,2,5] = C5.T
        curves[end_segment_mapper,2,6] = C6.T
        curves[end_segment_mapper,2,7] = C7.T
        curves[end_segment_mapper,2,8] = C8.T
        curves[end_segment_mapper,2,9] = C9.T
        curves[end_segment_mapper,2,10] = C10.T
        curves[end_segment_mapper,2,11] = C11.T
        start_segment_mapper = np.empty(data.start_xy.shape[0], dtype=valid_segments.dtype)
        start_segment_mapper[1:] = valid_segments
        start_segment_mapper[0] = False
        curves[start_segment_mapper,0,0] = D0.T
        curves[start_segment_mapper,0,1] = D1.T
        curves[start_segment_mapper,0,2] = D2.T
        curves[start_segment_mapper,0,3] = D3.T
        curves[start_segment_mapper,0,4] = D4.T
        curves[start_segment_mapper,0,5] = D5.T
        curves[start_segment_mapper,0,6] = D6.T
        curves[start_segment_mapper,0,7] = D7.T
        curves[start_segment_mapper,0,8] = D8.T
        curves[start_segment_mapper,0,9] = D9.T
        curves[start_segment_mapper,0,10] = D10.T
        curves[start_segment_mapper,0,11] = D11.T

        middle_start = np.array(data.start_xy.T, copy=True)
        middle_end = np.array(data.end_xy.T, copy=True)

        middle_start[:,start_segment_mapper] = D11
        middle_end[:,end_segment_mapper] = C0

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

        curve_lengths = np.full((data.start_xy.shape[0], 3), 0.0)
        curve_lengths[:,0] = np.linalg.norm(middle_start - data.start_xy.T, axis=0)
        curve_lengths[:,1] = np.linalg.norm(middle_end - middle_start, axis=0)
        curve_lengths[:,2] = np.linalg.norm(data.end_xy.T - middle_end, axis=0)
        curve_total_lengths = np.sum(curve_lengths, axis=1)

        curve_intervals = np.empty((data.start_xy.shape[0], 3))
        curve_intervals[:,0] = np.linspace(0, data.start_xy.shape[0]-1, data.start_xy.shape[0])
        curve_intervals[:,1] = curve_intervals[:,0] + \
                               np.divide(curve_lengths[:,0],
                                         curve_total_lengths,
                                         out=np.zeros_like(curve_lengths[:,0]),
                                         where=curve_total_lengths != 0)
        curve_intervals[:,2] = curve_intervals[:,1] + \
                               np.divide(curve_lengths[:,1],
                                         curve_total_lengths,
                                         out=np.zeros_like(curve_lengths[:,1]),
                                         where=curve_total_lengths != 0)
        curve_intervals = curve_intervals.reshape(3*curve_intervals.shape[0])
        curve_intervals2 = np.empty(np.sum(valid_curves) + 1)
        curve_intervals2[:-1] = curve_intervals[valid_curves]
        curve_intervals2[-1] = data.start_xy.shape[0]

        data.xy_spline = PHSpline(np.swapaxes(curves2[valid_curves], 0, 1), curve_intervals2)

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

        C0 = P1 - T0*(3*l + l_prime)
        C1 = P1 - 5*T0*l/2 - T0*l_prime
        C2 = P1 - 2*T0*l - T0*l_prime
        C3 = P1 - 3*T0*l/2 - T0*l_prime
        C4 = P1 - 97*T0*l/96 - T0*l_prime + T2*l/96
        C5 = P1 - 23*T0*l/42 - T0*l_prime + T2*l/21
        C6 = P1 - 25*T0*l/192 - T0*l_prime + 25*T2*l/192
        C7 = P1 + 51*T0*l/224 - T0*l_prime + 5*T1*l/1344 + 361*T2*l/1344
        C8 = P1 + 173*T0*l/336 - 255*T0*l_prime/256 + 5*T1*l/336 + T1*l_prime/256 + 2423*T2*l/5376
        C9 = P1 + 2591*T0*l/3584 - 251*T0*l_prime/256 + 127*T1*l/3584 + 5*T1*l_prime/256 + 577*T2*l/896
        C10 = P1 + 1521*T0*l/1792 - 121*T0*l_prime/128 + 121*T1*l/1792 + 7*T1*l_prime/128 + 363*T2*l/448
        C11 = P1 + 3215*T0*l/3584 - 227*T0*l_prime/256 + 415*T1*l/3584 + 29*T1*l_prime/256 + 825*T2*l/896

        D0 = P1 + 3215*T0*l/3584 - 227*T0*l_prime/256 + 415*T1*l/3584 + 29*T1*l_prime/256 + 825*T2*l/896
        D1 = P1 + 121*T0*l/128 - 53*T0*l_prime/64 + 21*T1*l/128 + 11*T1*l_prime/64 + 33*T2*l/32
        D2 = P1 + 469*T0*l/512 - 191*T0*l_prime/256 + 117*T1*l/512 + 65*T1*l_prime/256 + 139*T2*l/128
        D3 = P1 + 625*T0*l/768 - 163*T0*l_prime/256 + 241*T1*l/768 + 93*T1*l_prime/256 + 811*T2*l/768
        D4 = P1 + 125*T0*l/192 - T0*l_prime/2 + 41*T1*l/96 + T1*l_prime/2 + 59*T2*l/64
        D5 = P1 + 605*T0*l/1344 - 11*T0*l_prime/32 + 65*T1*l/112 + 21*T1*l_prime/32 + 925*T2*l/1344
        D6 = P1 + 55*T0*l/224 - 3*T0*l_prime/16 + 533*T1*l/672 + 13*T1*l_prime/16 + 67*T2*l/168
        D7 = P1 + 55*T0*l/672 - T0*l_prime/16 + 367*T1*l/336 + 15*T1*l_prime/16 + 31*T2*l/224
        D8 = P1 + 3*T1*l/2 + T1*l_prime
        D9 = P1 + 2*T1*l + T1*l_prime
        D10 = P1 + 5*T1*l/2 + T1*l_prime
        D11 = P1 + T1*(3*l + l_prime)

        curves = np.empty((data.start_xy.shape[0], 3, 12, 2))
        curves[:,:,0,0] = np.nan

        end_segment_mapper = np.empty(data.curve.shape[0], dtype=valid_segments.dtype)
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
        start_segment_mapper = np.empty(data.curve.shape[0], dtype=valid_segments.dtype)
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

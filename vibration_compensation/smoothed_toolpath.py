import numpy as np
from scipy.optimize import brentq
from .chandrupatla import chandrupatla
import math


class SmoothedToolpath(object):
    def __init__(self, maximum_error, start_xy, end_xy):
        self.start_xy = start_xy
        self.end_xy = end_xy
        self.maximum_error = maximum_error

        # Some lines have zero length, but let's still do the calculations and ignore them later
        with np.errstate(invalid="ignore"):
            vectors = self.end_xy.T - self.start_xy.T
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
        P1 = self.end_xy[:-1].T[:,valid_segments]
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

        self.curves = np.empty((12, 2, l.shape[0]))

        self.curves[0] = P1 - (3.0*l + l_prime) * T0
        self.curves[1] = P1 - (2.0*l + l_prime) * T0
        self.curves[2] = P1 - (l + l_prime) * T0
        self.curves[3] = P1 - l_prime * T0
        self.curves[4] = self.curves[3] + ((5.0 / 6.0) * l) * T0 + ((1.0 / 6.0) * l) * T2
        self.curves[5] = self.curves[4] + ((10.0 / 21.0) * l) * T0 + ((11.0 / 21.0) * l) * T2
        self.curves[6] = self.curves[5] + l * T2
        self.curves[7] = self.curves[6] + ((10.0 / 21.0) * l) * T1 + ((11.0 / 21.0) * l) * T2
        self.curves[8] = P1 + l_prime * T1
        self.curves[9] = P1 + (l + l_prime) * T1
        self.curves[10] = P1 + (2.0 * l + l_prime) * T1
        self.curves[11] = P1 + (3.0 * l + l_prime) * T1

        self.speed_coeffs = np.empty((11, l.shape[0]))

        end_segment_mapper = np.empty(self.start_xy.shape[0], dtype=valid_segments.dtype)
        end_segment_mapper[:-1] = valid_segments
        end_segment_mapper[-1] = False
        start_segment_mapper = np.empty(self.start_xy.shape[0], dtype=valid_segments.dtype)
        start_segment_mapper[1:] = valid_segments
        start_segment_mapper[0] = False

        if l.shape[0]:
            def eq_u0u3_u0v3():
                B3B4 = self.curves[4] - self.curves[3]
                lB3B4sq = np.einsum('ji,ji -> i', B3B4, B3B4)
                lsq = l*l
                a = 18.0 * 11.0 * lB3B4sq
                b = 13.0 * 11.0 * lsq
                c = 5*l
                return (a - b) / c

            u0u3_u0v3 = eq_u0u3_u0v3()

            self.speed_coeffs[0] = 11.0 * l
            self.speed_coeffs[1] = self.speed_coeffs[0]
            self.speed_coeffs[2] = self.speed_coeffs[0]
            self.speed_coeffs[3] = (1.0 / 6.0) * u0u3_u0v3 + (55.0 / 6.0)*l
            self.speed_coeffs[4] = (11.0 / 21.0) * u0u3_u0v3 + (110.0 / 21.0)*l
            self.speed_coeffs[5] = u0u3_u0v3
            self.speed_coeffs[6] = self.speed_coeffs[4]
            self.speed_coeffs[7] = self.speed_coeffs[3]
            self.speed_coeffs[8] = self.speed_coeffs[0]
            self.speed_coeffs[9] = self.speed_coeffs[0]
            self.speed_coeffs[10] = self.speed_coeffs[0]

            self.distance_coeffs = np.empty((12, self.speed_coeffs.shape[1]))

            comb = np.empty(12)

            self.distance_coeffs[0] = np.full(self.distance_coeffs.shape[1], 0.0)
            comb[0] = 1.0
            for i in range(11):
                self.distance_coeffs[i + 1] = self.speed_coeffs[i] + self.distance_coeffs[i]
                comb[i+1] = comb[i] * (1.0 * (11-i) / (i+1.0))

            self.distance_coeffs *= (comb / 11.0)[:,np.newaxis]
            self.coeffs = self.curves * comb[:,np.newaxis,np.newaxis]
        else:
            self.coeffs = None
            self.distance_coeffs = None

        middle_start = np.array(start_xy.T, copy=True)
        middle_end = np.array(end_xy.T, copy=True)

        middle_start[:,start_segment_mapper] = self.curves[11]
        middle_end[:,end_segment_mapper] = self.curves[0]

        num_lines = self.start_xy.shape[0]
        self.segment_start = np.empty((num_lines, 2))
        self.segment_number = np.empty((num_lines, 2), dtype=np.int)
        self.curve_number = np.full((num_lines, 2), -1, dtype=np.int)

        curve_indices = np.arange(length_a.shape[0], dtype=np.int)
        line_start = np.arange(self.start_xy.shape[0], dtype=np.float)
        self.segment_start[:,0] = line_start
        self.segment_number[:,0] = np.arange(self.start_xy.shape[0])
        self.segment_number[:,1] = np.full(num_lines, -1)

        self.segment_start[start_segment_mapper,0] = line_start[start_segment_mapper] + ((3*l + l_prime) / lengths[start_segment_mapper])

        self.segment_start[end_segment_mapper,1] = line_start[end_segment_mapper] + (1.0 - (3*l + l_prime) / lengths[end_segment_mapper])
        self.curve_number[end_segment_mapper,1] = curve_indices

        self.segment_start = self.segment_start.reshape(num_lines*2)
        self.segment_number = self.segment_number.reshape(num_lines*2)
        self.curve_number = self.curve_number.reshape(num_lines*2)

        to_keep = (self.curve_number != -1) | (self.segment_number != -1)
        self.segment_start = np.array(self.segment_start[to_keep], copy=True)
        self.segment_number = np.array(self.segment_number[to_keep], copy=True)
        self.curve_number = np.array(self.curve_number[to_keep], copy=True)
        self.segment_end = np.empty(self.segment_start.shape)
        self.segment_end[:-1] = self.segment_start[1:]
        self.segment_end[-1] = num_lines

        self.segment_lengths = np.empty(self.segment_start.shape[0])
        line_segments = self.segment_number != -1
        line_segment_numbers = self.segment_number[line_segments]
        lines = self.end_xy[line_segment_numbers] - self.start_xy[line_segment_numbers]
        scale_factor = self.segment_end[line_segments] - self.segment_start[line_segments]
        scaled_lines = lines * scale_factor[:,np.newaxis]
        self.segment_lengths[line_segments] = np.linalg.norm(scaled_lines, axis=1)
        self.segment_lengths[~line_segments] = np.sum(self.speed_coeffs, axis=0) / 11.0
        self.segment_distances = np.empty(self.segment_lengths.shape[0] + 1)
        self.segment_distances[1:] = np.cumsum(self.segment_lengths)
        self.segment_distances[0] = 0.0
        self._calculate_uv_coeffs()

    def __call__(self, x):
        return self._evaluate(x, False)

    def distance(self, x):
        return self._evaluate(x, True)

    def total_distance(self):
        return self.segment_distances[-1]

    def fixed_distances(self, start, end, num_steps):
        targets = np.linspace(start, end, num_steps)
        def f(t):
            return self.distance(t) - targets
        x0 = np.full(num_steps, 0.0)
        x1 = np.full(num_steps, self.start_xy.shape[0], dtype=np.float)
        t = chandrupatla(f, x0, x1)
        return t

    #The curve factor is reference curvature * reference speed * reference delta time
    def fixed_curvature_speeds(self, start, end, curve_factor):
        current = start
        ts = [start, current]
        unsigned_c = math.tan(0.5*curve_factor)
        while current < end:
            index = np.searchsorted(self.segment_start, current, side="right")
            index = index - 1
            index = np.atleast_1d(index)
            if self.segment_number[index] != -1:
                current = self.segment_end[index]
            else:
                uv0=self.uv[0,:,self.curve_number[index[0]]]
                uv5=self.uv[5,:,self.curve_number[index[0]]]
                uvdelta = uv5 - uv0
                curvature_sign = np.sign(uv0[0]*uvdelta[1] - uvdelta[0]*uv0[1])
                c = curvature_sign * unsigned_c

                prev = np.empty((1, 2))
                current_t = np.atleast_1d(current)
                self._evaluate_bernstein_common(prev, current_t, index, self.uv_coeffs)
                def f(t):
                    p = np.empty((1,2))
                    t = np.atleast_1d(t)
                    self._evaluate_bernstein_common(p, t, index, self.uv_coeffs)
                    return (c*prev[0,0] + prev[0,1])*p[0,0] - (prev[0,0] - c*prev[0,1])*p[0,1]
                try:
                    current = brentq(f, current, self.segment_end[index])
                except:
                    current = self.segment_end[index]
            ts.append(current)
        return np.array(ts, dtype=np.float)

    def _evaluate(self, x, is_distance=False):
        is_scalar = np.isscalar(x)
        x = np.atleast_1d(x)
        index = np.searchsorted(self.segment_start, x, side="right")
        index = index - 1
        if is_distance:
            ret = np.empty((x.shape[0]))
            self._evaluate_line_distance(ret, x, index)
            self._evaluate_bernstein_distance(ret, x, index)
            ret += self.segment_distances[index]
        else:
            ret = np.empty((x.shape[0], 2))
            self._evaluate_line(ret, x, index)
            self._evaluate_bernstein(ret, x, index)
        if is_scalar:
            return ret[0]
        else:
            return ret

    def _evaluate_line(self, res, x, index):
        lines = self.segment_number[index]
        line_filter = lines != -1
        filtered_indices = index[line_filter]
        filtered_lines = lines[line_filter]
        t_line = x[line_filter] - np.floor(self.segment_start[filtered_indices])
        dir = self.end_xy[filtered_lines] - self.start_xy[filtered_lines]
        res[line_filter] = self.start_xy[filtered_lines] + dir * t_line[:,np.newaxis]

    def _evaluate_line_distance(self, res, x, index):
        lines = self.segment_number[index]
        line_filter = lines != -1
        filtered_indices = index[line_filter]
        t_line = ((x[line_filter] - self.segment_start[filtered_indices]) /
            (self.segment_end[filtered_indices] - self.segment_start[filtered_indices]))
        res[line_filter] = self.segment_lengths[filtered_indices] * t_line

    def _evaluate_bernstein(self, res, x, index):
        if self.coeffs is not None:
            self._evaluate_bernstein_common(res, x, index, self.coeffs)

    def _evaluate_bernstein_distance(self, res, x, index):
        if self.distance_coeffs is not None:
            self._evaluate_bernstein_common(res, x, index, self.distance_coeffs)

    def _evaluate_bernstein_common(self, res, x, index, coeffs):
        curves = self.curve_number[index]
        curve_filter = curves != -1
        filtered_curves = curves[curve_filter]
        if filtered_curves.shape[0] == 0:
            return
        filtered_indices = index[curve_filter]
        mid_point = np.floor(self.segment_end[filtered_indices])
        on_next_segment = x[curve_filter] > mid_point
        t = np.empty_like(x[curve_filter])
        next_segment_index = filtered_indices[on_next_segment]
        t[on_next_segment] = (
                0.5 +
                0.5 * (x[curve_filter][on_next_segment] - mid_point[on_next_segment]) /
                (self.segment_end[next_segment_index] - mid_point[on_next_segment]))

        on_prev_segment = ~on_next_segment
        prev_segment_index = filtered_indices[on_prev_segment]
        t[on_prev_segment] = (
                0.5* (x[curve_filter][on_prev_segment] - self.segment_start[prev_segment_index]) /
                (mid_point[on_prev_segment] - self.segment_start[prev_segment_index]))

        res[curve_filter] = 0.0
        n = coeffs.shape[0] - 1
        t1 = 1.0 - t
        for k in range(n+1):
            a = t ** k * t1 ** (n - k)
            if len(coeffs.shape) > 2:
                a = a[...,np.newaxis]
            res[curve_filter] += coeffs[k,...,filtered_curves] * a


    def _calculate_uv_coeffs(self):
        d0 = np.empty(self.curves.shape[2], dtype=np.complex128)
        d5 = np.empty(self.curves.shape[2], dtype=np.complex128)
        d0.real = self.curves[1][0] - self.curves[0][0]
        d0.imag = self.curves[1][1] - self.curves[0][1]
        d5.real = self.curves[6][0] - self.curves[5][0]
        d5.imag = self.curves[6][1] - self.curves[5][1]
        K = 11

        d0 = d0 * K
        d5 = d5 * K

        w0 = np.sqrt(d0)
        w3 = d5 / w0

        self.uv = np.empty((6, 2, self.curves.shape[2]))
        self.uv[0][0] = w0.real
        self.uv[0][1] = w0.imag
        self.uv[1] = self.uv[0]
        self.uv[2] = self.uv[0]
        self.uv[3][0] = w3.real
        self.uv[3][1] = w3.imag
        self.uv[4] = self.uv[3]
        self.uv[5] = self.uv[3]
        comb = np.empty(6)
        comb[0] = 1.0
        for i in range(5):
            comb[i+1] = comb[i] * (1.0 * (5-i) / (i+1.0))
        self.uv_coeffs = self.uv.copy()
        self.uv_coeffs *= (comb / 5.0)[:,np.newaxis,np.newaxis]

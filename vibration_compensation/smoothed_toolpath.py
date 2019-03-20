import numpy as np


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

        end_segment_mapper = np.empty(self.start_xy.shape[0], dtype=valid_segments.dtype)
        end_segment_mapper[:-1] = valid_segments
        end_segment_mapper[-1] = False
        start_segment_mapper = np.empty(self.start_xy.shape[0], dtype=valid_segments.dtype)
        start_segment_mapper[1:] = valid_segments
        start_segment_mapper[0] = False

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

    def __call__(self, x):
        is_scalar = np.isscalar(x)
        x = np.atleast_1d(x)
        index = np.searchsorted(self.segment_start, x, side="right")
        index = index - 1
        ret = np.empty((x.shape[0], 2))
        self._evaluate_lines(ret, x, index)
        self._evaluate_bernstein(ret, x, index)
        if is_scalar:
            return ret[0]
        else:
            return ret

    def _evaluate_lines(self, res, x, index):
        lines = self.segment_number[index]
        line_filter = lines != -1
        filtered_indices = index[line_filter]
        filtered_lines = lines[line_filter]
        t_line = x[line_filter] - np.floor(self.segment_start[filtered_indices])
        dir = self.end_xy[filtered_lines] - self.start_xy[filtered_lines]
        res[line_filter] = self.start_xy[filtered_lines] + dir * t_line[:,np.newaxis]

    def _evaluate_bernstein(self, res, x, index):
        curves = self.curve_number[index]
        curve_filter = curves != -1
        filtered_curves = curves[curve_filter]
        if filtered_curves.shape[0] == 0:
            return
        filtered_indices = index[curve_filter]
        mid_point = np.floor(self.segment_end[filtered_indices])
        on_next_segment = x[curve_filter] > mid_point
        s = np.empty_like(x[curve_filter])
        next_segment_index = filtered_indices[on_next_segment]
        s[on_next_segment] = (
            0.5 +
            0.5 * (x[curve_filter][on_next_segment] - mid_point[on_next_segment]) /
            (self.segment_end[next_segment_index] - mid_point[on_next_segment]))

        on_prev_segment = ~on_next_segment
        prev_segment_index = filtered_indices[on_prev_segment]
        s[on_prev_segment] = (
            0.5* (x[curve_filter][on_prev_segment] - self.segment_start[prev_segment_index]) /
            (mid_point[on_prev_segment] - self.segment_start[prev_segment_index]))

        curves = self.curves[:,:,filtered_curves]
        res[curve_filter] = 0.0
        comb = 1.0
        k = 11
        s1 = 1.0 - s
        for j in range(k+1):
            res[curve_filter] += (comb * s**j * s1**(k-j) * curves[j]).T
            comb *= 1. * (k-j) / (j+1.)


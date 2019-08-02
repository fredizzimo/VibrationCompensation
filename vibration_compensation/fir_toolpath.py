from .data import Data
import numpy as np

class FirToolpath(object):
    def __init__(self, data: Data, maximum_acceleration, maximum_error, filter_times):
        self.maximum_error = maximum_error
        self.maximum_acceleration = maximum_acceleration
        self.xy = data.end_xy[1:].copy()
        self.f = data.f[1:].copy()

        lines = data.end_xy - data.start_xy
        self.length = np.linalg.norm(lines, axis=1)[1:]

        self.pulse_length = self.length / self.f
        end_times = np.cumsum(self.pulse_length)
        self.start_times = np.roll(end_times, 1)
        self.start_times[0] = 0
        self.total_time = end_times[-1] + np.sum(filter_times)
        assert len(filter_times) == 2
        self.filter_times = filter_times
        self.generate_fixed_coefficients()
        self.generate_all_coefficients()

    def generate_fixed_coefficients(self):
        c=[]
        # Avoid dividing by zero when the second filter time is zero
        T = self.filter_times
        if T[1] == 0:
            T = (0, T[0], 1)
        c.append(1 / T[1])
        c.append(c[0] / T[2])
        c.append(c[1] / 2)
        c.append(c[0] / 2)
        c.append(-T[2] * c[3])
        c.append(-c[2])
        c.append(T[1] + T[2])
        c.append(c[1] * c[6])
        c.append(-(T[1] ** 2 + T[2] ** 2) * c[2])
        c.append(2 * T[1])
        c.append(T[2] * c[9])
        c.append(-c[0])
        c.append(-c[1])
        c.append(c[1] / 6)
        c.append(-c[13])
        self.fixed_coefficients = c
        return c

    def generate_all_coefficients(self):
        self.s_coeffs = []
        self.v_coeffs = []
        self.a_coeffs = []
        for t_v, f in zip(np.nditer(self.pulse_length), np.nditer(self.f)):
            s_coeffs, v_coeffs, a_coeffs = self.generate_coefficients(f, t_v)
            s_coeffs.append(s_coeffs)
            v_coeffs.append(v_coeffs)
            a_coeffs.append(a_coeffs)

    def generate_coefficients(self, F, T_v):
        c = self.fixed_coefficients
        T = self.filter_times

        x0 = F * c[2]
        x1 = F * c[0]
        x2 = F * c[4]
        x3 = F * c[5]
        x4 = F * c[7]
        x5 = F * c[8]
        x6 = F * c[1]
        x7 = T_v * x6
        x8 = -x0 * (T_v ** 2 - c[10])
        x9 = F * c[11]
        x10 = F * c[3]
        x11 = x10 * (2 * T_v + T[1] + c[9])
        x12 = T_v + c[6]
        x13 = -x12 * x6
        x14 = x0 * x12 ** 2
        x15 = F * c[12]
        x16 = F * c[13]
        x17 = F * c[14]
        v_coeffs = [
            [x0, 0, 0],
            [0, x1, x2],
            [x3, x4, x5],
            [0, 0, F],
            [x3, x7, x8],
            [0, x9, x11],
            [x0, x13, x14],
            [0, 0, 0],
        ]
        a_coeffs = [
            [x6, 0],
            [0, x1],
            [x15, x4],
            [0, 0],
            [x15, x7],
            [0, x9],
            [x6, x13],
            [0, 0],
        ]
        s_coeffs = [
            [x16, 0, 0, 0],
            [0, x10, x2, 0],
            [x17, x0 * c[6], x5, 0],
            [0, 0, F, 0],
            [x17, T_v * x0, x8, 0],
            [0, -x10, x11, 0],
            [x16, -x0 * x12, x14, 0],
            [0, 0, 0, 0],
        ]

        return s_coeffs, v_coeffs, a_coeffs


    def v(self, t):
        pass




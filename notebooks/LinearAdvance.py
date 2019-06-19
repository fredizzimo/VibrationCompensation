#%%
%matplotlib inline
from IPython.display import display
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#%%
class Trapezoidal(object):
    def __init__(self, start_v, end_v, distance, max_v, max_a):
        self.start_v = start_v
        self.end_v = end_v
        self.distance = distance
        self.max_v = max_v
        self.max_a = max_a

        if distance * max_a > max_v**2.0 - (start_v**2.0 + end_v**2.0) / 2.0:
            self.cruise_v = max_v
            self.t_a = (max_v - start_v) / max_a
            self.t_d = (max_v - end_v) / max_a
            self.t = (
                distance / max_v +
                max_v / (2.0*max_a) * (1.0 - start_v / max_v)**2.0 +
                max_v / (2.0*max_a) * (1.0 - end_v / max_v)**2.0
            )
        else:
            self.cruise_v = math.sqrt(distance*max_a + (start_v**2.0 + end_v**2.0) / 2.0)
            if self.cruise_v > max_v:
                self.cruise_v = max_v
            self.t_a = (self.cruise_v - start_v) / max_a
            self.t_d = (self.cruise_v - end_v) / max_a
            self.t = self.t_a + self.t_d

    def v(self, t):
        if t < self.t_a:
            return self.start_v + (self.cruise_v - self.start_v) / self.t_a * t
        elif t <= self.t - self.t_d:
            return self.cruise_v
        elif t <= self.t:
            return self.end_v + (self.cruise_v - self.end_v) / self.t_d * (self.t - t)
        else:
            return self.end_v

    def a(self, t):
        if t < self.t_a:
            return self.max_a
        elif t <= self.t - self.t_d:
            return 0
        elif t <= self.t:
            return -self.max_a
        else:
            return 0

    def x(self, t):
        if t < self.t_a:
            return (
               self.start_v * t +
               (self.cruise_v - self.start_v) / (2.0*self.t_a) * t**2
            )
        elif t <= self.t - self.t_d:
            return self.start_v*self.t_a / 2.0 + self.cruise_v * (t - self.t_a / 2.0)
        elif t <= self.t:
            return (
                self.distance -
                self.end_v * (self.t - t) -
                (self.cruise_v - self.end_v) / (2.0*self.t_d) * (self.t - t)**2.0
            )
        else:
            return self.distance

#%%
s_v = sp.IndexedBase("v")
s_a = sp.IndexedBase("a")
s_t = sp.IndexedBase("t")
s_vstart = s_v["start"]
s_vend = s_v["end"]
s_vcruise = s_v["cruise"]
s_vmax = s_v["max"]
s_d = sp.Symbol("d")
s_amax = s_a["max"]
s_ta = s_t["acc"]
s_td = s_t["dcc"]
s_tt = s_t["tot"]

eq_vcruise = sp.Eq(s_vcruise, sp.Piecewise(
    (s_vmax, s_d * s_amax > s_vmax**2 - (s_vstart**2 + s_vend**2) / 2),
    (sp.Max(s_vmax,sp.sqrt(s_d * s_amax + (s_vstart**2 + s_vend**2) / 2)), True)
))
eq_ta = sp.Eq(s_ta, (s_vcruise - s_vstart) / s_amax)
eq_td = sp.Eq(s_td, (s_vcruise - s_vend) / s_amax)

display(eq_vcruise)
display(eq_ta)
display(eq_td)
print(eq_vcruise)

#%%
f_x = sp.Function("x", real=True)
f_e = sp.Function("e", real=True) 
s_t = sp.Symbol("t", nonnegative=True, real=True)
s_k = sp.Symbol("k", nonnegative=True, real=True)

eq_lin_advance = sp.Eq(f_e(s_t), f_x(s_t) + s_k * sp.Derivative(f_x(s_t),s_t))
display(eq_lin_advance)
lin_advance_ics = {f_x(0): 0}
# f_x(s_t).diff().subs(s_t, 0): 0}
display(lin_advance_ics)
eq_x=sp.dsolve(eq_lin_advance, ics=lin_advance_ics).doit()
display(eq_x)

#%%
s_a = sp.Symbol("a", real=True)
eq_lin_acc = eq_x.subs(f_e(s_t), 0.5*s_a*s_t**2).doit().simplify()
display(eq_lin_acc)
l_eq_lin_acc = sp.lambdify([s_t, s_a, s_k], eq_lin_acc.rhs, modules="numpy")

xs = np.linspace(0, 1, 1000)
ys = [l_eq_lin_acc(x, 1000, 0.5) for x in xs]
es = [0.5*1000.0*x**2 for x in xs]

plt.plot(xs, ys)
plt.plot(xs, es)

#%%
s_l = sp.Symbol("l", real=True)
eq_a_t = sp.Eq(s_l, eq_lin_acc.rhs)
print(eq_a_t)
display(eq_a_t)
display(sp.solve(eq_a_t, [s_t]))
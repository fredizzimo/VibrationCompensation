# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 1

import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from types import SimpleNamespace
from vibration_compensation.symbols import Symbols
import sympy as sp
from IPython.display import display, Math, Markdown
# %aimport vibration_compensation.symbols

sp.init_printing(order="grevlex")


def print_equation(lhs, rhs):
    display(Markdown(sp.latex(sp.Eq(lhs, rhs), mode="equation")))


s = Symbols()

r = SimpleNamespace()

# # Trapezoidal Trajectory

# The acceleration $a_{t}$ of a trapezoidal trajectory is defined as

r.a_t = sp.Piecewise(
    (s.a_max, s.t <= s.t_ta),
    (0, s.t <= s.t_ta + s.t_tc),
    (-s.a_max, s.t <= s.t_ta + s.t_tc + s.t_td),
    (0, True)
)
print_equation(s.a_t, r.a_t)

# Integration gives the velocity $v_{t}$ and position $x_{t}$

v = s.v_s + sp.integrate(r.a_t, s.t)
v = sp.simplify(v)
r.v_t = v
print_equation(s.v_t, r.v_t)


x = sp.integrate(r.v_t, s.t)
x = sp.simplify(x)
r.x_t = x
print_equation(s.x_t, r.x_t)

# The time $t_{amax}$ it takes to accelerate from $v_{s}$ to $v_{e}$ with the maximum acceleration $a_{max}$

t_amax = sp.Abs(s.v_e - s.v_s) / s.a_max
print_equation(s.t_amax, t_amax)

# The trajectory is possible when the travel distance $\Delta_{x}$ is long enough.

eq = s.d >= sp.Piecewise(
    (s.v_s * s.t_amax + sp.S(1) / 2 * s.a_max * s.t_amax**2, s.v_e - s.v_s >= 0),
    (s.v_s * s.t_amax - sp.S(1) / 2 * s.a_max * s.t_amax**2, True)
)
display(eq)
eq = eq.rhs.subs(s.t_amax, t_amax.rewrite(sp.Piecewise))
eq = sp.piecewise_fold(eq)
eq = sp.simplify(eq)
display(eq)
eq = sp.Abs(eq.args[0][0], evaluate=False)
display(eq)
eq = eq.cancel().ratsimp()
display(eq)

# Klipper defines $\Delta_{v^2}$ (*delta_v2*), which simplifies the calculations

delta_v2 = sp.Abs(s.v_e**2 - s.v_s**2)
print_equation(s.delta_v2, delta_v2)

# We can substitute into (TODO: insert here), which gives

delta_v2_dist_amax = 2*s.d*s.a_max
sp.Le(s.delta_v2, delta_v2_dist_amax)

# Then we need to determine if there's a cruise phase. That's true when we can accelerate from $v_{s}$ to $v_{max}$ with $a_{max}$ and then decelerate to $v_{e}$ within the travel distance.
#
# If that's the case, then we have

t_ta = sp.Eq(s.t_ta, (s.v_max - s.v_s) / s.a_max)
display(t_ta)

# And

t_td = sp.Eq(s.t_td, (s.v_max - s.v_e) / s.a_max)
display(t_td)

# That means that we have a cruise phase when

eq = s.d > r.x_t.args[2][0]
display(eq)
eq = eq.rhs.subs(s.t_tc, 0)
display(eq)
eq = eq.subs(s.t, s.t_ta + s.t_td).simplify()
display(eq)
eq = eq.subs(s.t_ta, t_ta.rhs).subs(s.t_td, t_td.rhs)
display(eq)
display(eq.simplify().ratsimp())

# The cruise time $t_{tc}$ can be solved

eq = r.x_t.args[2][0]
eq = eq.subs(s.t, s.t_ta + s.t_tc + s.t_td).simplify()
display(eq)
eq = eq.subs(s.t_ta, t_ta.rhs).subs(s.t_td, t_td.rhs).simplify()
display(eq)
res = sp.solve(sp.Eq(s.d, eq), s.t_tc)[0]
res = res.expand().ratsimp()
display(res)

res2 = res + t_ta.rhs + t_td.rhs
res2 = res2.simplify()
display(res2)
display(Markdown(sp.latex(res2, order="lex", mode="equation")))
display(res2.ratsimp())

hello = -8 + 7 * s.a_max  + 5 * s.v_max**2 
display(hello)
display(Markdown(sp.latex(hello, order="grlex", mode="equation")))
display(Markdown(sp.latex(hello, order="grevlex", mode="equation")))
display(Markdown(sp.latex(hello, order="lex", mode="equation")))
display(Markdown(sp.latex(hello, order="ilex", mode="equation")))

from sympy.parsing.latex import parse_latex
test = parse_latex(r"2 + x").subs(sp.Symbol("x"), sp.Symbol("x", positive=True))
assert test == 2 + sp.Symbol("x", positive=True)

(s.v_s + (s.v_e + delta_v2_dist_amax)) / 2

# Given the cruise speed of the trajectory, $v_{c}$ we can calculate the acceleration and deceleration times

t_ta = sp.Eq(s.t_ta, (s.v_c - s.v_s) / s.a_max)
display(t_ta)

t_td = sp.Eq(s.t_td, (s.v_c - s.v_e) / s.a_max)
display(t_td)

# And the cruise time can be be calculated by requiring the distance traveled at $t = t_{ta} + t_{tc} + t_{td}$ to be $\Delta_{x}$ and solving for $t_{tc}$ 

eq = sp.Eq(s.d, r.x_t.args[2][0].subs(s.t, s.t_ta + s.t_tc + s.t_td))
eq = eq.simplify()
display(eq)
t_tc = sp.solve(eq, s.t_tc)[0].ratsimp()
display(t_tc)

# This can be simplified by substituting $t_{ta}$ and $t_{td}$

display(res.subs(s.t_ta, t_ta.lhs).subs(s.t_td, t_td.rhs).simplify())

# Since the distance to move determines how long we can accelerate and decelerate, we can calculate the distance moved in each part using timeless kinematic equation

v_f = sp.Symbol("v_f")
v_i = sp.Symbol("v_i")
a = sp.Symbol("a")
x = sp.Symbol("x")
timeless = sp.Eq(v_f**2, v_i**2 + 2*a*x)
display(timeless)
timeless_x = sp.solve(timeless, x)[0]
display(sp.Eq(x, timeless_x))

delta_xta = (s.v_tc**2 - s.v_s**2) / (2*s.a_max)
display(sp.Eq(s.delta_xta, delta_xta))

delta_xtd = (s.v_e**2 - s.v_tc**2) / (-2*s.a_max)
delta_xtd = delta_xtd.simplify()
display(sp.Eq(s.delta_xtd, delta_xtd))

# The cruise distance is the distance that's left

delta_xtc = s.d - s.delta_xta - s.delta_xtd
display(sp.Eq(s.delta_xtc, delta_xtc))

# $$\Delta_{xtc} = \Delta_{x} - \Delta_{xta} - \Delta_{xtd}$$

# If there's no cruise phase, then the cruise distance $\Delta_{xtc} = 0$ and the squared cruise speed $v_{tc}^2$ can be determined.

v_tc = sp.Min(sp.solve(s.d - delta_xta - delta_xtd, s.v_tc**2)[0], s.v_max**2)
display(v_tc)

# \begin{equation*}
# \Delta_{x} - \Delta_{xta} - \Delta_{xtd} = 0 \\
# \Delta_{x} - \frac{v_{tc}^2 - v_{s}^2}{2 a_{max}} - \frac{v_{tc}^2 - v_{e}^2}{2 a_{max}}  = 0 \\
# v_{tc}^2 = \Delta_{x}a_{max} + \frac{v_{s}^2 + v_{e}^2}{2}
# \end{equation*}

# Otherwise the maximum speed is reached and the squared cruise speed is $v_{tc}^2 = v_{max}^2$.
# So the two cases can be combined into
# \begin{equation}
# v_{tc}^2 = \min\left(\Delta_{x}a_{max} + \frac{v_{s}^2 + v_{e}^2}{2},{v_{max}^2}\right)
# \end{equation}

# The above can be used to calculate the exact distances $\Delta_{xta}$, $\Delta_{xtc}$ and  $\Delta_{xtd}$
#
# Finally, the acceleration and deceleration times can be calculated from those distances

t_ta = s.delta_xta / ((s.v_s + s.v_tc) / 2)
t_ta = t_ta.simplify()
t_tc = s.delta_xtc / (s.v_tc)
t_tc = t_tc.simplify()
t_td = s.delta_xtd / ((s.v_tc + s.v_e) / 2)
t_td = t_td.simplify()
display(t_ta)
display(t_tc)
display(t_td)

# \begin{equation}
# t_{ta} = \frac{2\Delta_{xta}}{v_{s} + v_{tc}} \\
# t_{tc} = \frac{\Delta_{xtc}}{v_{tc}} \\
# t_{td} = \frac{2\Delta_{xtd}}{v_{tc} + v_{e}}
# \end{equation}

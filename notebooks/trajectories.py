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

sp.init_printing()


def print_equation(lhs, rhs):
    display(Markdown(sp.latex(sp.Eq(lhs, rhs), mode="equation")))


s = Symbols()

r = SimpleNamespace()

# # Trapezoidal Trajectory

# The acceleration ($a_{t}$) of a trapezoidal trajectory is defined as

r.a_t = sp.Piecewise(
    (s.a_max, s.t <= s.t_ta),
    (0, s.t <= s.t_ta + s.t_tc),
    (-s.a_max, s.t <= s.t_ta + s.t_tc + s.t_td),
    (0, True)
)
print_equation(s.a_t, r.a_t)

# Integration gives the velocity ($v_{t}$) and position ($x_{t}$)

v = sp.integrate(r.a_t, s.t)
v = sp.simplify(v)
r.v_t = v
print_equation(s.v_t, r.v_t)


x = sp.integrate(r.v_t, s.t)
x = sp.simplify(x)
r.x_t = x
print_equation(s.x_t, r.x_t)



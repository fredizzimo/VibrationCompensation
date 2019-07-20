# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sympy as sp

s_s = sp.Symbol("s")
s_i = sp.Symbol("i", integer=True)
s_Ti = sp.IndexedBase("T")[s_i]
s_Mi = sp.IndexedBase("M")[s_i]
display(a*b)
f_Mi = sp.Eq(sp.Function("M_i")(s_s), (1 / s_Ti) * ((1 - sp.E**(-s_s*s_Ti)) / s_s))
display(f_Mi)

s_t = sp.Symbol("t")
res = sp.inverse_laplace_transform(f_Mi.rhs, s_s, s_t)
f_mi = sp.Eq(sp.Function("m_i")(s_t), res)
display(f_mi)



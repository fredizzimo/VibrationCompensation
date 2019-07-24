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

# +
s_t = sp.Symbol("t", nonnegative=True)

s_s = sp.Symbol("s")
s_i = sp.Symbol("i", integer=True)
s_Ti = sp.IndexedBase("T")[s_i]
s_F = sp.Symbol("F")
f_v_in = sp.Function("v_in")
s_t_in = sp.Symbol("t_in")
s_d = sp.Symbol("d")

f_Mi = sp.Function("M_i")
f_mi = sp.Function("m_i")
# -

display(a*b)
eq_Mi = sp.Eq(s_Mi(s_s), (1 / s_Ti) * ((1 - sp.E**(-s_s*s_Ti)) / s_s))
display(eq_Mi)

res = sp.inverse_laplace_transform(eq_Mi.rhs, s_s, s_t)
eq_mi = sp.Eq(f_mi(s_t), res)
display(eq_mi)

# $\theta$ is the Heaviside function

eq_v_in = sp.Eq(f_v_in(s_t),
      sp.Piecewise(
          (s_F, s_t <= s_t_in),
          (0, s_t > s_t_in)
      ))
display(eq_v_in)

eq_v_in = sp.Eq(f_v_in(s_t), s_F * (sp.Heaviside(s_t) - sp.Heaviside(s_t - s_t_in)))
display(eq_v_in)

eq_t_in = sp.Eq(s_t_in, s_d / s_F)
display(eq_t_in)

# +
s_tau = sp.Symbol("tau")
def convolute(f, g):
    f = f.subs(s_t, s_t - s_tau)
    g = g.subs(s_t, s_tau)
    a = sp.Integral(f.lhs*g.lhs, (s_tau, 0, s_t))
    b = sp.Integral(f.rhs*g.rhs, (s_tau, 0, s_t))
    return (a, b)

res = convolute(eq_mi, eq_v_in)
display(res[0])
display(res[1])
res = res[1].simplify()
display(res)
# Heaviside(s_tau) is always 1
res = res.replace(sp.Heaviside(s_tau), 1)
display(res)
# and so is this
res = res.replace(sp.Heaviside(s_t - s_tau), 1).simplify()
display(res)

#res = res[1].rewrite(sp.Piecewise, H0=1).simplify()
#display(res)
#res = res[1].doit()
#display(res)
#res = res.simplify()
#display(res)
#res = res.rewrite(sp.Piecewise, H0=1)
#display(res)
#res = sp.piecewise_fold(res)
#display(res)
#display(sp.piecewise_fold(res.rewrite(sp.Piecewise)))
#res = res[1].doit()
#display(res)
#display(res)
#display(res)
#display(sp.piecewise_fold(res).simplify())


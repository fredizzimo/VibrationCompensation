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
s_T = sp.IndexedBase("T")
s_Ti = s_T[s_i]
s_F = sp.Symbol("F")
f_v_in = sp.Function("v_in")
s_t_in = sp.Symbol("t_in")
s_d = sp.Symbol("d")
s_T_v = sp.Symbol("T_v")

f_Mi = sp.Function("M_i")
f_mi = sp.Function("m_i")
f_v = sp.Function("v")
f_a = sp.Function("a")
f_s = sp.Function("s")

f_s_i = [sp.Function("s_%i" % (i,)) for i in range(8)]
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

# -

eq_v = sp.Eq(f_v(s_t), sp.Piecewise(
    ((s_F * s_t**2) / (2 * s_T[1] * s_T[2]), s_t < s_T[2]),
    ((s_F * s_T[2]) / (2 * s_T[1]) + (s_F / s_T[1])*(s_t - s_T[2]), s_t < s_T[1]),
    (s_F - ((s_F * (s_T[1] + s_T[2] - s_t)**2)) / (2*s_T[1]*s_T[2]), s_t < s_T[1] + s_T[2]),
    (s_F, s_t < s_T_v),
    (s_F - (s_F * (s_t - s_T_v)**2) / (2 * s_T[1] * s_T[2]), s_t < s_T_v + s_T[2]),
    (s_F - (s_F * s_T[2]) / (2 * s_T[1]) - (s_F * (s_t - s_T_v - s_T[2])) / s_T[1], s_t < s_T_v + s_T[1]),
    ((s_F * (s_T_v + s_T[1] + s_T[2] - s_t)**2) / (2 * s_T[1] * s_T[2]), s_t < s_T_v + s_T[1] + s_T[2]),
    (0, True)
))
display(eq_v)

eq_a = sp.Eq(f_a(s_t), sp.diff(eq_v.rhs, s_t))
display(eq_a)


def integrate(f):
    lower = 0
    current = 0
    res = []
    for p in f.args:
        if p[1] == True:
            upper = s_t
        else:
            upper = sp.solve(p[1], s_t).args[1]
        integ_res = sp.integrate(p[0], (s_t, 0, s_t)) 
        integ_res = integ_res.simplify()
        res.append((integ_res, p[1]))
        #current = current + sp.integrate(p[0], (s_t, lower, upper)) 
        lower = upper
    return sp.Piecewise(*res)
eq_s = sp.Eq(f_s(s_t), integrate(eq_v.rhs))
display(eq_s)
#sp.solve(eq_v.rhs.args[4][1], s_t)

eq_f_s_i = [sp.Eq(f_s_i[i](s_t), sp.integrate(p[0], (s_t, 0, s_t)).simplify()) for i, p in enumerate(eq_v.rhs.args)]
for eq in eq_f_s_i:
    display(eq)


# +
def generate_s():
    lower = 0
    current = 0
    res = []
    for i in range(len(eq_v.rhs.args)):
        v = eq_v.rhs.args[i]
        if v[1] == True:
            upper = s_t
        else:
            upper = sp.solve(v[1], s_t).args[1]
        if i == 0:
            res.append((f_s_i[0](s_t), v[1]))
        else:
            res.append((f_s_i[i-1](lower) + f_s_i[i](s_t) - f_s_i[i](lower), v[1]))
        lower = upper
    # The final distance is known
    res[-1] = (s_F*s_T_v, res[-1][1])
    return sp.Eq(f_s(s_t), sp.Piecewise(*res))

eq_f_s = generate_s()

display(eq_f_s)

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
import sys

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
    ((s_F * s_t**2) / (2 * s_T[0] * s_T[1]), s_t < s_T[1]),
    ((s_F * s_T[1]) / (2 * s_T[0]) + (s_F / s_T[0])*(s_t - s_T[1]), s_t < s_T[0]),
    (s_F - ((s_F * (s_T[0] + s_T[1] - s_t)**2)) / (2*s_T[0]*s_T[1]), s_t < s_T[0] + s_T[1]),
    (s_F, s_t < s_T_v),
    (s_F - (s_F * (s_t - s_T_v)**2) / (2 * s_T[0] * s_T[1]), s_t < s_T_v + s_T[1]),
    (s_F - (s_F * s_T[1]) / (2 * s_T[0]) - (s_F * (s_t - s_T_v - s_T[1])) / s_T[0], s_t < s_T_v + s_T[0]),
    ((s_F * (s_T_v + s_T[0] + s_T[1] - s_t)**2) / (2 * s_T[0] * s_T[1]), s_t < s_T_v + s_T[0] + s_T[1]),
    (0, True)
))
display(eq_v)

eq_a = sp.Eq(f_a(s_t), sp.diff(eq_v.rhs, s_t))
display(eq_a)

eq_f_s_i = [sp.Eq(f_s_i[i](s_t), sp.integrate(p[0], (s_t, 0, s_t)).expand().factor()) for i, p in enumerate(eq_v.rhs.args)]
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

# -

# Generate code for evaluating the trajectories

# +
# First convert to standard polynomials
def convert_expr_to_poly(expr):
    poly = sp.Poly(expr, s_t)
    coeffs = poly.all_coeffs()
    coeffs = [c.expand().factor() for c in coeffs] 
    poly = sp.Poly(coeffs, s_t)
    return poly.as_expr()

def convert_to_polynomial(eq):
    res = []
    for p in eq.rhs.args:
        res.append((convert_expr_to_poly(p[0]), p[1]))
    return sp.Eq(eq.lhs, sp.Piecewise(*res))


eq_v_polynomial = convert_to_polynomial(eq_v)
eq_a_polynomial = convert_to_polynomial(eq_a)
eq_f_s_i_polynomial = [sp.Eq(e.lhs, convert_expr_to_poly(e.rhs)) for e in eq_f_s_i]

# We are interested in just the coefficents
def get_coeffs(p, degree):
    poly = sp.Poly(p, s_t)
    poly_degree = poly.degree(s_t)
    if poly_degree == -sp.oo:
        poly_degree = 0
    return ([0] * (degree - poly_degree)) + poly.all_coeffs()

v_coeffs = sp.Matrix([get_coeffs(p[0], 2) for p in eq_v_polynomial.rhs.args])
a_coeffs = sp.Matrix([get_coeffs(p[0], 1) for p in eq_a_polynomial.rhs.args])
s_coeffs = sp.Matrix([get_coeffs(p.rhs, 3) for p in eq_f_s_i_polynomial])

# The feedrate is always a factor, so remove it
def divide_by_f(m):
    res = m / s_F
    return res.applyfunc(sp.factor)

v_coeffs_without_f = divide_by_f(v_coeffs)
a_coeffs_without_f = divide_by_f(a_coeffs)
s_coeffs_without_f = divide_by_f(s_coeffs)

# Then generate all common subexpressions that doesn't involve T_v, since that is also feedrate indepedent.
# That way we only need to call these once
def fixed_coefficients():
    c = sp.IndexedBase("c")
    for i in range(sys.maxsize):
        yield c[i]

fixed_coefficients = sp.cse([v_coeffs_without_f, a_coeffs_without_f, s_coeffs_without_f], ignore=(s_T_v,), symbols=fixed_coefficients())

res = fixed_coefficients[1]
res = [(r * s_F).applyfunc(sp.factor) for r in res]
    
# Extract all the remaining subexpressions, we have to re-calculate this everytime the speed changes
feedrate_dependent_coefficients = sp.cse(res)

# Print the result, so it can be copied
for r in fixed_coefficients[0]:
    print("c.append(%s)" % r[1])
    
for r in feedrate_dependent_coefficients[0]:
    print("%s = %s" % r)
    
def assign_array(name, matrix):
    l = matrix.tolist()
    print("%s = [" % (name,))
    for r in l:
        print("    %s," % (r,))
    print("]")

assign_array("v_coeffs", feedrate_dependent_coefficients[1][0])
assign_array("a_coeffs", feedrate_dependent_coefficients[1][1])
assign_array("s_coeffs", feedrate_dependent_coefficients[1][2])
# -



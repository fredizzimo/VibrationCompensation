# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import sympy as sym

# %%
sym.init_printing()

# %%
k, k_0, c, s, th_0 = sym.symbols("kappa, kappa_0, c, s, theta_0")
T_hat, i_hat, j_hat = sym.symbols(r"\hat{T}, \hat{i}, \hat{j}")
x_0, y_0, t = sym.symbols("x_0, y_0, t")
v, v_x, v_y = sym.symbols("v, v_x, v_y")
a, a_x, a_y = sym.symbols("a, a_x, a_y")
j, j_x, j_y = sym.symbols("j, j_x, j_y")

# %%
f_k_s = sym.Function("kappa")(s)
eq_k_s = sym.Eq(f_k_s, k_0 + c*s)
display(eq_k_s)

# %%
f_theta_s = sym.Function("theta")(s)
eq_theta_s = sym.Eq(f_theta_s, th_0 + k_0 * s + (c*s**2) / 2)
display(eq_theta_s)

# %%
f_T_hat_s = sym.Function(r"\hat{T}")(s)
eq_t_hat1 = sym.Eq(f_T_hat_s, sym.cos(f_theta_s) * i_hat + sym.sin(f_theta_s) * j_hat)
display(eq_t_hat1)
eq_t_hat2 = sym.Eq(eq_t_hat1.lhs, eq_t_hat1.rhs.subs(f_theta_s, eq_theta_s.rhs))
display(eq_t_hat2)

# %%
f_r_vec_s = sym.Function(r"\overrightarrow{r}")(s)
f_x_s = sym.Function("x")(s)
f_y_s = sym.Function("y")(s)
eq_r_vec_s = sym.Eq(f_r_vec_s, f_x_s * i_hat + f_y_s * j_hat)
display(eq_r_vec_s)

# %%
eq_x_s = sym.Eq(f_x_s, x_0 + sym.Integral(sym.cos(th_0 + k_0*t + (c*t**2 / 2)), (t, 0, s)))
display(eq_x_s)
eq_y_s = sym.Eq(f_y_s, y_0 + sym.Integral(sym.sin(th_0 + k_0*t + (c*t**2 / 2)), (t, 0, s)))
display(eq_y_s)

# %%
eq_v_x = sym.Eq(v_x, v * sym.cos(f_theta_s))
display(eq_v_x)
eq_v_y = sym.Eq(v_y, v * sym.sin(f_theta_s))
display(eq_v_y)

# %%
eq_a_x = sym.Eq(a_x, a * sym.cos(f_theta_s) - v**2 * k_0 * sym.sin(f_theta_s))
display(eq_a_x)
eq_a_y = sym.Eq(a_y, a * sym.sin(f_theta_s) + v**2 * k_0 * sym.cos(f_theta_s))
display(eq_a_y)

# %%
eq_j_x = sym.Eq(j_x,
    j * sym.cos(f_theta_s)
    - 3 * v * a * f_k_s * sym.sin(f_theta_s)
    - v**3 * f_k_s**2 * sym.cos(f_theta_s)
    - c * v**3 * sym.sin(f_theta_s)
)
display(eq_j_x)
eq_j_y = sym.Eq(j_y,
    j * sym.sin(f_theta_s)
    + 3 * v * a * f_k_s * sym.cos(f_theta_s)
    - v**3 * f_k_s**2 * sym.sin(f_theta_s)
    + c * v**3 * sym.cos(f_theta_s)
)
display(eq_j_x)

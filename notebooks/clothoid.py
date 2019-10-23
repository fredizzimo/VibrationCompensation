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

# %%
eq_k_s = sym.Eq(sym.Function("kappa")(s), k_0 + c*s)
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

# %%
eq_y_s = sym.Eq(f_y_s, y_0 + sym.Integral(sym.sin(th_0 + k_0*t + (c*t**2 / 2)), (t, 0, s)))
display(eq_y_s)

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

# %%
eq_k_s = sym.Eq(sym.Function("kappa")(s), k_0 + c*s)
display(eq_k_s)

# %%
f_theta_s = sym.Function("theta")(s)
eq_theta_s = sym.Eq(f_theta_s, th_0 + k_0 * s + (c*s**2) / 2)
display(eq_theta_s)

# %%
f_T_hat = sym.Function(r"\hat{T}")(s)
eq_t_hat1 = sym.Eq(f_T_hat, sym.cos(f_theta_s) * i_hat + sym.sin(f_theta_s) * j_hat)
display(eq_t_hat1)
eq_t_hat2 = sym.Eq(eq_t_hat1.lhs, eq_t_hat2.rhs.subs(f_theta_s, eq_theta_s.rhs))
display(eq_t_hat2)

# %%

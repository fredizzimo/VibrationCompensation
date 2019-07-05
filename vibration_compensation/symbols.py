import sympy as sp


class Symbols:
    t = sp.Symbol("t", nonnegative=True)
    a_max = sp.Symbol("a_max", positive=True)
    v_max = sp.Symbol("v_max", positive=True)

    d = sp.Symbol(r"\Delta_{x}", nonnegative=True)
    v_s = sp.Symbol("v_s", nonnegative=True)
    v_e = sp.Symbol("v_e", nonnegative=True)
    v_c = sp.Symbol("v_c", nonnegative=True)
    delta_v2 = sp.Symbol(r"\Delta_{v^2}", nonnegative=True)

    t_ta = sp.Symbol("t_ta", nonnegative=True)
    t_tc = sp.Symbol("t_tc", nonnegative=True)
    t_td = sp.Symbol("t_td", nonnegative=True)
    delta_xta = sp.Symbol(r"\Delta_{xta}")
    delta_xtc = sp.Symbol(r"\Delta_{xtc}")
    delta_xtd = sp.Symbol(r"\Delta_{xtd}")
    v_tc = sp.Symbol("v_tc")

    a_t = sp.Symbol("a_t")
    v_t = sp.Symbol("v_t")
    x_t = sp.Symbol("x_t")

    t_amax = sp.Symbol("t_amax")

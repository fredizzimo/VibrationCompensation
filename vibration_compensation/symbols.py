import sympy as sp


class Symbols:
    t = sp.Symbol("t", nonnegative=True)
    a_max = sp.Symbol("a_max", positive=True)
    v_max = sp.Symbol("v_max", positive=True)

    t_ta = sp.Symbol("t_ta", nonnegative=True)
    t_tc = sp.Symbol("t_tc", nonnegative=True)
    t_td = sp.Symbol("t_td", nonnegative=True)

    a_t = sp.Symbol("a_t")
    v_t = sp.Symbol("v_t")
    x_t = sp.Symbol("x_t")

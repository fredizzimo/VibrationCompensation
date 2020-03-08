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
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
import numpy as np
from math import ceil, sqrt
import sympy as sp


# %%
def calculate_segments(segments):
    linear_advance=0.05
    dt = 0.001
    num_t = [int(ceil(segment_time / dt)) for _,_,_,_,_,segment_time in segments]
    total_num_t = np.sum(num_t)
    res = np.empty((8, total_num_t))
    index = 0
    t = np.nan
    x = np.nan
    v = np.nan
    a = np.nan
    j = np.nan
    
    for (start_t, start_x, start_v, start_a, start_j, segment_time), n in zip(segments, num_t):
        ts = np.linspace(0.0, segment_time, n, endpoint=True, dtype=np.float)
        if not np.isnan(start_t):
            t = start_t
        if not np.isnan(start_x):
            x = start_x
        if not np.isnan(start_v):
            v = start_v 
        if not np.isnan(start_a):
            a = start_a
        if not np.isnan(start_j):
            j = start_j
        
        if n > 0:
            res[1,index:index+n] = x + v * ts + 0.5 * a * ts**2 + j * ts**3 / 6.0
            res[2,index:index+n] = v + a * ts + 0.5 * j * ts**2
            res[3,index:index+n] = a + j * ts
            res[4,index:index+n] = np.full(n, j)
            res[5,index:index+n] = res[1,index:index+n] + linear_advance * res[2,index:index+n]
            res[6,index:index+n] = v + a*ts + 0.5*j*ts**2 + linear_advance * (a + j*ts)
            res[7,index:index+n] = a + j*ts + linear_advance * j
            ts += t
            res[0,index:index+n]=ts
            t = res[0,index+n-1]
            x = res[1,index+n-1]
            v = res[2,index+n-1]
            a = res[3,index+n-1]
            j = res[4,index+n-1]
            
            index+=n
    
    return res

def graph_segments(segments):
    result = calculate_segments(segments)
    fig = go.Figure()
    x_color = DEFAULT_PLOTLY_COLORS[0]
    v_color = DEFAULT_PLOTLY_COLORS[1]
    a_color = DEFAULT_PLOTLY_COLORS[2]
    j_color = DEFAULT_PLOTLY_COLORS[3]
    
    fig.add_trace(go.Scatter(
        x=result[0], y=result[1], name="position",
        yaxis="y1",
        legendgroup="position",
        line=go.scatter.Line(color=x_color)))
    fig.add_trace(go.Scatter(
        x=result[0], y=result[2], name="velocity",
        yaxis="y2",
        legendgroup="velocity",
        line=go.scatter.Line(color=v_color)))
    fig.add_trace(go.Scatter(
        x=result[0], y=result[3], name="acceleration",
        yaxis="y3",
        legendgroup="acceleration",
        line=go.scatter.Line(color=a_color)))
    fig.add_trace(go.Scatter(
        x=result[0], y=result[4], name="jerk",
        yaxis="y4",
        legendgroup="jerk",
        line=go.scatter.Line(color=j_color)))
    fig.add_trace(go.Scatter(
        x=result[0], y=result[5], name="Linear Advance x",
        yaxis="y1",
        legendgroup="position",
        line=go.scatter.Line(color=x_color, dash="dash")))
    fig.add_trace(go.Scatter(
        x=result[0], y=result[6], name="Linear Advance v",
        yaxis="y2",
        legendgroup="velocity",
        line=go.scatter.Line(color=v_color, dash="dash")))
    fig.add_trace(go.Scatter(
        x=result[0], y=result[7], name="Linear Advance a",
        yaxis="y3",
        legendgroup="acceleration",
        line=go.scatter.Line(color=a_color, dash="dash")))
    tickfont = go.layout.yaxis.Tickfont(size=8)
    titlefont = go.layout.yaxis.title.Font(size=8)
    fig.update_layout(
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="time(s)"
            ),
            domain=[0.18,1]
        ),
        yaxis=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="position(mm)"
            ),
            titlefont=titlefont,
            tickfont=tickfont,
            side="left",
            position=0.18
        ),
        yaxis2=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="velocity(mm/s)"
            ),
            titlefont=titlefont,
            tickfont=tickfont,
            anchor="free",
            overlaying="y",
            side="left",
            position=0.12
        ),
        yaxis3=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="acceleration(mm/s^2)"
            ),
            titlefont=titlefont,
            tickfont=tickfont,
            anchor="free",
            overlaying="y",
            side="left",
            position=0.06
        ),
        yaxis4=go.layout.YAxis(
            title=go.layout.yaxis.Title(
                text="jerk(mm/s^3)"
            ),
            titlefont=titlefont,
            tickfont=tickfont,
            anchor="free",
            overlaying="y",
            side="left",
            position=0.0
        )
    )
    fig.show()
    
def graph_trapezoidal_t(start_v, accel, accel_t, cruise_t, decel_t):
    segments = [
        (0, 0, start_v, accel, 0, accel_t),
        (np.nan, np.nan, np.nan, 0, 0, cruise_t),
        (np.nan, np.nan, np.nan, -accel, 0, decel_t),
    ]
    graph_segments(segments)
    
def graph_jerk_t(start_v, jerk, t1, t2, t3, t4, t5, t6, t7):
    segments = [
        (0, 0, start_v, 0, jerk, t1),
        (np.nan, np.nan, np.nan, np.nan, 0, t2),
        (np.nan, np.nan, np.nan, np.nan, -jerk, t3),
        (np.nan, np.nan, np.nan, np.nan, 0, t4),
        (np.nan, np.nan, np.nan, np.nan, -jerk, t5),
        (np.nan, np.nan, np.nan, np.nan, 0, t6),
        (np.nan, np.nan, np.nan, np.nan, jerk, t7),
    ]
    graph_segments(segments)



# %%
graph_trapezoidal_t(0, 1000, 0.1, 0.1, 0.1)

# %%
graph_jerk_t(0, 10000, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1)


# %%
def generate_trapezoidal(start_v, end_v, distance, max_v, max_a):
    assert max_v >= start_v
    assert max_v >= end_v
    start_v2 = start_v**2
    end_v2 = end_v**2
    max_v2 = max_v**2
    cruise_v2 = distance * max_a + 0.5 * (start_v2 + end_v2)
    cruise_v2 = min(max_v2, cruise_v2)
    assert cruise_v2 >= end_v2
    half_inv_accel = 0.5 / max_a
    accel_d = (cruise_v2 - start_v2) * half_inv_accel
    decel_d = (cruise_v2 - end_v2) * half_inv_accel
    cruise_d = distance - accel_d - decel_d
    cruise_v = sqrt(cruise_v2)
    # Determine time spent in each portion of move (time is the
    # distance divided by average velocity)
    accel_t = accel_d / ((start_v + cruise_v) * 0.5)
    cruise_t = cruise_d / cruise_v
    decel_t = decel_d / ((end_v + cruise_v) * 0.5)
    return (
        (0, 0, start_v, max_a, 0, accel_t),
        (np.nan, np.nan, np.nan, 0, 0, cruise_t),
        (np.nan, np.nan, np.nan, -max_a, 0, decel_t),
    )

def graph_trapezoidal(start_v, end_v, distance, max_v, max_a):
    graph_segments(generate_trapezoidal(start_v, end_v, distance, max_v, max_a))



# %%
graph_trapezoidal(0, 10, 20, 100, 1000)


# %%
def generate_jerk(start_v, end_v, distance, max_v, max_a, max_j):
    jerk_t = max_a / max_j
    delta_distance = start_v * (jerk_t / 2) + end_v * (jerk_t / 2)
    trapezoidal = generate_trapezoidal(start_v, end_v, distance-delta_distance, max_v, max_a)
    print("start")
    print(trapezoidal)
    graph_segments(trapezoidal)
    print("end")
    accel_t = trapezoidal[0][5]
    cruise_t = trapezoidal[1][5]
    decel_t = trapezoidal[2][5]
    t2 = accel_t - jerk_t
    t4 = cruise_t - jerk_t
    t6 = decel_t - jerk_t
    tolerance = 1e-10
    if t4 <= -tolerance:
        max_v = sqrt((max_a*jerk_t)**2.0 + 4.0*max_a*distance + 2.0*start_v**2.0 + 2.0*end_v**2.0)
        max_v -= jerk_t * max_a
        max_v /= 2.0
        return generate_jerk(start_v, end_v, distance, max_v, max_a, max_j)
    if t2 <= -tolerance:
        max_a = sqrt(max_j / max_v)
        print(max_a)
        return generate_jerk(start_v, end_v, distance, max_v, max_a, max_j)
    
    def remove_empty(segments):
        return [s for s in segments if abs(0.0 - s[5]) > tolerance]
    
    return remove_empty((
        (0, 0, start_v, 0, max_j, jerk_t),
        (np.nan, np.nan, np.nan, np.nan, 0, t2),
        (np.nan, np.nan, np.nan, np.nan, -max_j, jerk_t),
        (np.nan, np.nan, np.nan, np.nan, 0, t4),
        (np.nan, np.nan, np.nan, np.nan, -max_j, jerk_t),
        (np.nan, np.nan, np.nan, np.nan, 0, t6),
        (np.nan, np.nan, np.nan, np.nan, max_j, jerk_t),
    ))

def graph_jerk(start_v, end_v, distance, max_v, max_a, max_j):
    res = generate_jerk(start_v, end_v, distance, max_v, max_a, max_j)
    graph_segments(res)

def graph_trapezoidal_and_jerk(start_v, end_v, distance, max_v, max_a, max_j):
    graph_trapezoidal(start_v, end_v, distance, max_v, max_a)
    graph_jerk(start_v, end_v, distance, max_v, max_a, max_j)

#graph_trapezoidal_and_jerk(95, 0, 20, 100, 1000, 100000)


# %%
#No adaptation
graph_trapezoidal_and_jerk(0, 0, 20, 100, 1000, 100000)

# %%
#Type 2
graph_trapezoidal_and_jerk(0, 0, 10.5, 100, 1000, 100000)


# %%
def adjust_cruise_speed_formula():
    v_s, v_c, v_e = sp.symbols("v_s v_c v_e")
    d = sp.symbols("d")
    t_a, t_c, t_d = sp.symbols("t_a t_c t_d")
    a_a, a_d = sp.symbols("a_a a_d")

    eq_d = sp.Eq(d, sp.together((v_c**2 - v_s**2) / (2 * a_a)) + v_c * t_c + sp.together((v_c**2 - v_e**2) / (2 * a_d)))
    
    display(eq_d)
    eq_v_c = sp.Eq(v_c, sp.together(sp.solve(eq_d, v_c)[0]))
    eq_v_c = eq_v_c.factor()
    display(eq_v_c)
    display(eq_v_c.rhs.args[2].args[0].args[1].args[0].simplify())
    subexpr = sp.cse(eq_v_c.rhs, optimizations="basic")
    for expr in subexpr[0]:
        display(sp.Eq(expr[0], expr[1]))
    display(subexpr[1][0].collect(("x0", "x4", "x3")))
    print(subexpr[1][0])

adjust_cruise_speed_formula()


# %%
def calculate_jerk(ts, v_start, j):
    jerks = [
        j,
        0,
        -j,
        0,
        -j,
        0,
        j
    ]
    x = 0
    v = v_start
    a = 0
    for t, j in zip(ts, jerks):
        x += v * t + (a * t**2) / 2 + (j * t**3) / 6
        v += a * t + (j * t**2) / 2
        a += j * t
    return x.simplify(), v.simplify()


# %%
def adjust_distance_formula():
    v_s, v_c, v_e = sp.symbols("v_s v_c v_e")
    d, d_jerk, delta_d = sp.symbols(r"d d_jerk {\Delta}d")
    t_a, t_c, t_d = sp.symbols("t_a t_c t_d")
    a_a, a_d = sp.symbols("a_a a_d")
    j = sp.symbols("j")
    
    ts = [
        a_a / j,
        t_a - (a_a / j),
        a_a / j,
        t_c - (a_a / j),
        a_d / j,
        t_d - (a_d / j),
        a_d / j
    ]

    eq_d_jerk = sp.Eq(d_jerk, calculate_jerk(ts, v_s, j)[0])
    display(eq_d_jerk.factor())
    eq_delta_d = sp.Eq(delta_d, eq_d_jerk.lhs - d)
    display(eq_delta_d)
    eq_delta_d = sp.Eq(delta_d, eq_d_jerk.rhs - d)
    eq_delta_d = eq_delta_d.factor()
    display(eq_delta_d)
    
    eq_t_a = sp.Eq(t_a, (v_c-v_s) / a_a)
    eq_t_d = sp.Eq(t_d, (v_c-v_e) / a_d)
    display(eq_t_a)
    display(eq_t_d)
    
    
    eq_d = sp.Eq(d, sp.together((v_c**2 - v_s**2) / (2 * a_a)) + v_c * t_c + sp.together((v_c**2 - v_e**2) / (2 * a_d)))
    display(eq_d)
    eq_t_c = sp.Eq(t_c, sp.solve(eq_d, t_c)[0])
    display(eq_t_c)
    
    eq_delta_d = eq_delta_d.subs(eq_t_a.lhs, eq_t_a.rhs)
    eq_delta_d = eq_delta_d.subs(eq_t_d.lhs, eq_t_d.rhs)
    eq_delta_d = eq_delta_d.subs(eq_t_c.lhs, eq_t_c.rhs)
    eq_delta_d = eq_delta_d.expand().simplify()
    display(eq_delta_d)

adjust_distance_formula()


# %%
def full_jerk_reached_max_a_formula():
    a_max = sp.symbols("a_max")
    j = sp.symbols("j")
    d = sp.symbols("d")
    v_s, v_e = sp.symbols("v_s v_e")
    t_a, t_c, t_d = sp.symbols("t_a t_c t_d")
    
    t_a = a_max / j 
    t_d = a_max / j
    
    ts = [
        t_a,
        t_c,
        t_d,
        0,
        0,
        0,
        0
    ]
    eq_d, eq_v = calculate_jerk(ts, v_s, j)
    eq_d = sp.Eq(d, eq_d)
    eq_v = sp.Eq(v_e, eq_v)
    display(eq_d)
    display(eq_v)
    eq_t_c = sp.Eq(t_c, sp.solve(eq_v, t_c)[0])
    display(eq_t_c)
    
    eq_d = sp.Eq(d, eq_d.rhs.subs(t_c, eq_t_c.rhs))
    display(eq_d.simplify())

full_jerk_reached_max_a_formula()


# %%
def full_jerk_max_a_not_reached_formula():
    a_max = sp.symbols("a_max")
    j = sp.symbols("j")
    d = sp.symbols("d")
    v_s, v_e = sp.symbols("v_s v_e")
    t_a, t_c, t_d = sp.symbols("t_a t_c t_d")
    
    t_a = a_max / j 
    t_d = a_max / j
    
    ts = [
        t_a,
        0,
        t_d,
        0,
        0,
        0,
        0
    ]
    eq_d, eq_v = calculate_jerk(ts, v_s, j)
    eq_d = sp.Eq(d, eq_d)
    eq_v = sp.Eq(v_e, eq_v)
    display(eq_d)
    display(eq_v)
    
    eq_a_max = sp.Eq(a_max, sp.solve(eq_v, a_max)[1])
    display(eq_a_max)
    
    eq_d = eq_d.subs(a_max, eq_a_max.rhs).simplify()
    display(eq_d)
    

full_jerk_max_a_not_reached_formula()


# %%
def max_end_v_max_a_not_reached_formula():
    j = sp.symbols("j", real=True)
    d = sp.symbols("d", postive=True)
    v_s, v_e = sp.symbols("v_s v_e", positive=True)
    t = sp.symbols("t", positive=True)
    p, q = sp.symbols("p q")
    
    ts = [
        t/2,
        0,
        t/2,
        0,
        0,
        0,
        0
    ]
    eq_d, eq_v = calculate_jerk(ts, v_s, j)
    eq_d = sp.Eq(d, eq_d)
    eq_v = sp.Eq(v_e, eq_v)
    
    display(eq_d)
    display(eq_v)
    
    eq_t = sp.Eq(t**2, sp.solve(eq_v, t**2)[0])
    display(eq_t)
    
    eq_d = eq_d.subs(t**2, eq_t.rhs).simplify()
    display(eq_d)
    eq_d = sp.Eq(eq_d.lhs**2, eq_d.rhs**2)
    display(eq_d)
    eq_d = sp.Eq(eq_d.rhs - eq_d.lhs, 0)
    display(eq_d)
    eq_d_diff = eq_d.lhs.diff(v_e).simplify()
    display(eq_d_diff)
    
    eq_newton = eq_d.lhs / eq_d_diff
    eq_newton = eq_newton.simplify()
    display(eq_newton)

max_end_v_max_a_not_reached_formula()


# %%
def max_end_v_max_a_reached_formula():
    a_max = sp.symbols("a_max")
    j = sp.symbols("j")
    d = sp.symbols("d")
    d_min = sp.symbols("d_min")
    v_s, v_e = sp.symbols("v_s v_e")
    t_a, t_c, t_d = sp.symbols("t_a t_c t_d")
    
    t_a = a_max / j 
    t_d = a_max / j
    
    ts = [
        t_a,
        t_c,
        t_d,
        0,
        0,
        0,
        0
    ]
    eq_d, eq_v = calculate_jerk(ts, v_s, j)
    eq_d = sp.Eq(d, eq_d)
    eq_v = sp.Eq(v_e, eq_v)
    display(eq_d)
    display(eq_v)
    
    eq_d_min = sp.Eq(d_min, eq_d.rhs.subs(t_c, 0))
    display(eq_d_min)
    
    eq_t_c = sp.Eq(t_c, sp.solve(eq_v, t_c)[0])
    display(eq_t_c)
    eq_d = sp.Eq(d, eq_d.rhs.subs(t_c, eq_t_c.rhs).simplify())
    display(eq_d)
    eq_v_e = sp.Eq(v_e, sp.solve(eq_d, v_e)[0].simplify())
    display(eq_v_e)
    
    temp = eq_v_e.rhs.args[2].args[0].args[0]
    display(temp)
    poly = sp.poly(temp, j)
    display(poly.nth(0))
    eq_v_e = eq_v_e.replace(temp, sp.horner(poly))
    display(eq_v_e)
    
max_end_v_max_a_reached_formula()


# %%
def full_acc_allowed_no_const_acc_formula():
    a_max = sp.symbols("a_max")
    j = sp.symbols("j")
    d = sp.symbols("d")
    d_full = sp.symbols("d_full")
    v_s, v_e = sp.symbols("v_s v_e")
    t_a = sp.symbols("t_a")
    
    ts = [
        t_a,
        0,
        0,
        0,
        0,
        0,
        0
    ]
    eq_d, eq_v = calculate_jerk(ts, v_s, j)
    eq_d = sp.Eq(d, eq_d)
    eq_v = sp.Eq(v_e, eq_v)
    display(eq_d)
    display(eq_v)
    eq_d_full = sp.Eq(d_full, eq_d.rhs.subs(t_a, a_max / j).simplify())
    display(eq_d_full)
    eq_t_a = sp.Eq(t_a, sp.solve(eq_v, t_a)[1])
    display(eq_t_a)
    eq_d = eq_d.subs(t_a, eq_t_a.rhs).simplify()
    display(eq_d)
    
    
full_acc_allowed_no_const_acc_formula()


# %%
def full_acc_allowed_const_acc_formula():
    a_max = sp.symbols("a_max")
    j = sp.symbols("j")
    d = sp.symbols("d")
    d_min = sp.symbols("d_min")
    v_s, v_e = sp.symbols("v_s v_e")
    t_a, t_c = sp.symbols("t_a t_c")
    
    t_a = a_max / j
    
    ts = [
        t_a,
        t_c,
        0,
        0,
        0,
        0,
        0
    ]
    eq_d, eq_v = calculate_jerk(ts, v_s, j)
    eq_d = sp.Eq(d, eq_d)
    eq_v = sp.Eq(v_e, eq_v)
    display(eq_d)
    display(eq_v)
    eq_t_c = sp.Eq(t_c, sp.solve(eq_v, t_c)[0])
    display(eq_t_c)
    eq_d = eq_d.subs(t_c, eq_t_c.rhs).simplify()
    
    display(eq_d)
    
    
full_acc_allowed_const_acc_formula()


# %%
def short_distance_same_speed_formula():
    a_max = sp.symbols("a_max, a_new")
    j = sp.symbols("j")
    d = sp.symbols("d")
    v = sp.symbols("v")
    v_s, v_c, v_e = sp.symbols("v_s v_c v_e")
    t_a, t_c, t_d = sp.symbols("t_a t_c t_d")
    
    ts = [
        t_a,
        0,
        t_a,
        0,
        t_d,
        0,
        t_d
    ]
    eq_d_full, eq_v_full = calculate_jerk(ts, v_s, j)
    eq_d_full = sp.Eq(d, eq_d_full)
    eq_v_full = sp.Eq(v_e, eq_v_full)
    eq_d = sp.Eq(d, eq_d_full.rhs.subs(t_d, t_a))
    eq_v = sp.Eq(v, eq_d.rhs.diff(t_a))
    display(eq_d)
    display(eq_v)
    eq_v_c =sp.Eq(v_c, eq_v_full.rhs.subs(t_d, 0))
    display(eq_v_c)
    eq_t_a = sp.Eq(t_a, sp.solve(eq_v_c, t_a)[1])
    display(eq_t_a)
    
    
    

short_distance_same_speed_formula()


# %%
def adaptation_formulas():
    d, d_adj = sp.symbols("d d_adj")
    d_a, d_ad, d_c, d_d = sp.symbols("d_a d_ad d_c d_d")
    v_s, v_c, v_e = sp.symbols("v_s v_c v_e")
    a_a, a_d = sp.symbols("a_a a_d")
    j = sp.symbols("j")
    t_a = sp.symbols("t_a")
    eq_dist = sp.Eq(d_a + d_ad + d_c + d_d,d_adj)
    display(eq_dist)
    eq_d_a = sp.Eq(d_a, (v_c**2 - v_s**2) / (2*a_a))
    display(eq_d_a)
    eq_d_ad = sp.Eq(d_ad, v_c*(a_a/j))
    display(eq_d_ad)
    eq_d_d = sp.Eq(d_d, (v_c**2 - v_e**2) / (2*a_d))
    display(eq_d_d)
    eq_d_adj = sp.Eq(d_adj, d - (a_a*(v_s - v_c) + a_d*(v_e + v_c)) / (2*j))
    display(eq_d_adj)
    eq = eq_dist.subs(d_adj, eq_d_adj.rhs).subs(d_a, eq_d_a.rhs).subs(d_ad, eq_d_ad.rhs).subs(d_d, eq_d_d.rhs)
    display(eq)
    eq_no_adapt = sp.Eq(d_c, sp.solve(eq, d_c)[0])
    display(eq_no_adapt)
    
    eq_type_II = eq_no_adapt.subs(d_c, 0)
    eq_type_II_poly = sp.Poly(eq_type_II.rhs, v_c)
    eq_type_II = sp.Eq(eq_type_II_poly.as_expr(), 0)
    display(eq_type_II)
    
    eq_new_accel = sp.Eq(a_a, sp.sqrt(j*(v_c - v_s)))
    display(eq_new_accel)
    eq_new_decel = sp.Eq(a_d, sp.sqrt(j*(v_c - v_e)))
    display(eq_new_decel)
    eq_v_c = sp.Eq(v_c, sp.solve(eq_new_accel, v_c)[0])
    display(eq_v_c)
    
    eq = eq_dist.subs(d_c, 0)
    display(eq)
    eq = eq.subs(d_adj, eq_d_adj.rhs).subs(d_a, eq_d_a.rhs).subs(d_ad, eq_d_ad.rhs).subs(d_d, eq_d_d.rhs)
    display(eq)
    eq = eq.subs(v_c, eq_v_c.rhs)
    display(eq)
    poly = sp.Poly(eq, a_a)
    eq = sp.Eq(poly.as_expr(), 0)
    display(eq)
    exprs = sp.cse(poly.all_coeffs())
    for e in exprs[0]:
        print("%s = %s" % e)
    print("a = %s" % (exprs[1][0],))
    print("b = %s" % (exprs[1][1],))
    print("c = %s" % (exprs[1][2],))
    print("d = %s" % (exprs[1][3],))
    print("e = %s" % (exprs[1][4],))

adaptation_formulas()


# %%
def optimize_jerk_profile(distance, start_v, max_v, end_v, accel, jerk):
    import numpy as np
    from scipy.optimize import minimize

    def calculate_move(jerk_t):
        x = 0
        v = start_v
        a = 0
        j = jerk
        jerk_multipliers = [
            1,
            0,
            -1,
            0,
            -1,
            0,
            1
        ]

        distances = []
        speeds = []
        accs = []
        jerks = []

        for i, segment in enumerate(jerk_t):
            t = segment
            if t:
                j = jerk * jerk_multipliers[i]
                x += v * t + 0.5 * a * t**2 + j * t**3 / 6.0
                v += a * t + 0.5 * j * t**2
                a += j * t
            distances.append(x)
            speeds.append(v)
            accs.append(a)
            jerks.append(j)
        return distances, speeds, accs, jerks

    def f(jerk_t):
        return np.sum(jerk_t)

    def cons_distance(jerk_t):
        distances, _, _, _ = calculate_move(jerk_t)
        return distances[-1] - distance

    def cons_speed(jerk_t):
        _, speeds, _, _ = calculate_move(jerk_t)
        speeds = np.array(speeds)
        return max_v - speeds

    def cons_endv(jerk_t):
        _, speeds, _, _ = calculate_move(jerk_t)
        return end_v - speeds[-1]

    def cons_accel(jerk_t):
        _, _, accs, _ = calculate_move(jerk_t)
        accs = np.array(accs)
        accs = np.abs(accs)
        return accel - accs

    def cons_enda(jerk_t):
        _, _, accs, _ = calculate_move(jerk_t)
        return 0.0 - accs[-1]


    cons = (
        {"type": "eq", "fun": cons_distance},
        {"type": "eq", "fun": cons_endv},
        {"type": "eq", "fun": cons_enda},
        {"type": "ineq", "fun": cons_speed},
        {"type": "ineq", "fun": cons_accel}
    )
    res = minimize(f, [0,0,0,0,0,0,0], bounds=[(0,None)]*7, constraints=cons, options={"ftol": 1e-12})
    print(res)
    if res.success:
        graph_jerk_t(start_v, jerk, *res.x)


# %%
optimize_jerk_profile(6.6, 95, 100, 30, 1000, 100000)

# %%
optimize_jerk_profile(6.4, 95, 100, 30, 1000, 100000)

# %%
t = 0.06
graph_jerk_t(30, 100000, t, 0, t, 0, 0, 0, 0)

# %%

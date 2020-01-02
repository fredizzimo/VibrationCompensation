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
    dt = 0.001
    num_t = [int(ceil(segment_time / dt)) for _,_,_,_,_,segment_time in segments]
    total_num_t = np.sum(num_t)
    res = np.empty((5, total_num_t))
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
    a_max = sp.symbols("a_max")

    eq_d = sp.Eq(d, sp.together((v_c**2 - v_s**2) / (2 * a_max)) + v_c * t_c + sp.together((v_c**2 - v_e**2) / (2 * a_max)))
    
    display(eq_d)
    eq_v_c = sp.Eq(v_c, sp.together(sp.solve(eq_d, v_c)[1]))
    display(eq_v_c)

adjust_cruise_speed_formula()


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
    jerks = [
        j,
        0,
        -j,
        0,
        -j,
        0,
        j
    ]
    def calculate_jerk():
        x = 0
        v = v_s
        a = 0
        for t, j in zip(ts, jerks):
            x += v * t + (a * t**2) / 2 + (j * t**3) / 6
            v += a * t + (j * t**2) / 2
            a += j * t
        return x.simplify()
    eq_d_jerk = sp.Eq(d_jerk, calculate_jerk())
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

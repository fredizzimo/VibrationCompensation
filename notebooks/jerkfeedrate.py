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

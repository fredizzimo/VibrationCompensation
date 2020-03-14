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
import sympy as sp


# %%
def fixed_extruder_speed():
    x_ts = sp.symbols("x_ts")
    v_e, v_ts, v_t = sp.symbols("v_e v_ts v_t")
    
    c = sp.symbols("c")
    t = sp.symbols("t")
    fx_t = sp.Function("x_t")
    eq_advance = sp.Eq(v_e*t, fx_t(t) + c*sp.Derivative(fx_t(t), t))
    display(eq_advance)
    #initial_state = {fx_t(0): x_ts}
    initial_state = {fx_t(t).diff(t).subs(t, 0): v_ts}
    eq_toolhead = sp.dsolve(eq_advance, fx_t(t), ics=initial_state)
    display(eq_toolhead.simplify())
    

fixed_extruder_speed()


# %%
def decelerate_from_to():
    x_xs, x_xe, x_te = sp.symbols("x_xs x_xe x_te")
    v_es, v_ee, v_ts, v_te, v_ec = sp.symbols("v_es v_ee v_ts v_te v_ec")
    a_e = sp.symbols("a_e", positive=True)
    c = sp.symbols("c")
    t, t_d, t_c, t_a = sp.symbols("t t_d t_c t_a")
    fx_t = sp.Function("x_t")
    display("deceleration part")
    eq_advance1 = sp.Eq(x_xs + v_es*t - (a_e*t**2)/2, fx_t(t) + c*sp.Derivative(fx_t(t), t))
    display(eq_advance1)
    eq_toolhead1 = sp.dsolve(eq_advance1, fx_t(t), ics={fx_t(0): 0})
    eq_toolhead1 = eq_toolhead1.simplify()
    display(eq_toolhead1)
    
    display("cruise part")
    eq_advance2 = sp.Eq(eq_advance1.lhs.subs(t, t_d) + v_ec*t, fx_t(t) + c*sp.Derivative(fx_t(t), t))
    display(eq_advance2)
    eq_toolhead2 = sp.dsolve(eq_advance2, fx_t(t), ics={fx_t(0): eq_toolhead1.rhs.subs(t, t_d)})
    eq_toolhead2 = eq_toolhead2.simplify()
    display(eq_toolhead2)
    
    display("acceleration part")
    eq_advance3 = sp.Eq(eq_advance2.lhs.subs(t, t_c) + (a_e*t**2)/2, fx_t(t) + c*sp.Derivative(fx_t(t), t))
    display(eq_advance3)
    eq_toolhead3 = sp.dsolve(eq_advance3, fx_t(t), ics={fx_t(0): eq_toolhead2.rhs.subs(t, t_c)})
    eq_toolhead3 = eq_toolhead3.simplify()
    display(eq_toolhead3)
    
    display(eq_advance3.lhs.subs(t, t_a))
    eq_x_xe = sp.Eq(x_xe, eq_advance3.lhs.subs(t, t_a))
    eq_x_xe = eq_x_xe.simplify()
    display(eq_x_xe)
    eq_x_te = sp.Eq(x_te, eq_toolhead3.rhs.subs(t, t_a))
    eq_x_te = eq_x_te.simplify()
    display(eq_x_te)
    
    eq_t_a = sp.Eq(t_a, (v_ee - v_ec)/a_e)
    display(eq_t_a)
    eq_t_d = sp.Eq(t_d, (v_es - v_ec)/a_e)
    display(eq_t_d)
    eq_x_xs = sp.Eq(x_xs, c*v_ts)
    display(eq_x_xs)
    eq_x_xe2 = sp.Eq(x_xe, x_te + c*v_te)
    display(eq_x_xe2)
    
    eq_x_xe = eq_x_xe.subs(eq_t_a.lhs, eq_t_a.rhs)
    eq_x_xe = eq_x_xe.subs(eq_t_d.lhs, eq_t_d.rhs)
    eq_x_xe = eq_x_xe.subs(eq_x_xs.lhs, eq_x_xs.rhs)
    eq_x_xe = eq_x_xe.subs(x_xe, eq_x_xe2.rhs)
    eq_x_xe = eq_x_xe.simplify()
    display(eq_x_xe)
    
    eq_v_ec = sp.Eq(v_ec, sp.solve(eq_x_xe, v_ec)[0].simplify())
    display(eq_v_ec)
    eq_t_c = sp.Eq(t_c, sp.solve(eq_x_xe, t_c)[0])
    display(eq_t_c)
    
    eq_x_te = eq_x_te.subs(eq_t_a.lhs, eq_t_a.rhs)
    eq_x_te = eq_x_te.subs(eq_t_d.lhs, eq_t_d.rhs)
    eq_x_te = eq_x_te.subs(eq_x_xs.lhs, eq_x_xs.rhs)
    eq_x_te = eq_x_te.simplify()
    display(eq_x_te)
    
    #eq_x_te = eq_x_te.subs(eq_v_ec.lhs, eq_v_ec.rhs).simplify()
    #display(eq_x_te)
    
    #eq_t_c = sp.Eq(t_c, sp.solve(eq_x_te, t_c)[0])
    #display(eq_t_c)
    
    eq_x_te = eq_x_te.subs(eq_t_c.lhs, eq_t_c.rhs).simplify()
    display(eq_x_te)
    
    eq_v_ec = sp.Eq(v_ec, sp.solve(eq_x_te, v_ec)[0])
    display(eq_v_ec)
    

decelerate_from_to()


# %%
def graph_pressure(ts, toolhead_y, toolhead_v, toolhead_a, toolhead_j, extruder_y, extruder_v, extruder_a, extruder_j):
    fig = go.Figure()
    x_color = DEFAULT_PLOTLY_COLORS[0]
    v_color = DEFAULT_PLOTLY_COLORS[1]
    a_color = DEFAULT_PLOTLY_COLORS[2]
    j_color = DEFAULT_PLOTLY_COLORS[3]
    
    fig.add_trace(go.Scatter(
        x=ts, y=toolhead_y, name="position",
        yaxis="y1",
        legendgroup="position",
        line=go.scatter.Line(color=x_color)))
    fig.add_trace(go.Scatter(
        x=ts, y=toolhead_v, name="velocity",
        yaxis="y2",
        legendgroup="velocity",
        line=go.scatter.Line(color=v_color)))
    fig.add_trace(go.Scatter(
        x=ts, y=toolhead_a, name="acceleration",
        yaxis="y3",
        legendgroup="acceleration",
        line=go.scatter.Line(color=a_color)))
    fig.add_trace(go.Scatter(
        x=ts, y=toolhead_j, name="jerk",
        yaxis="y4",
        legendgroup="jerk",
        line=go.scatter.Line(color=j_color)))
    fig.add_trace(go.Scatter(
        x=ts, y=extruder_y, name="Linear Advance x",
        yaxis="y1",
        legendgroup="position",
        line=go.scatter.Line(color=x_color, dash="dash")))
    fig.add_trace(go.Scatter(
        x=ts, y=extruder_v, name="Linear Advance v",
        yaxis="y2",
        legendgroup="velocity",
        line=go.scatter.Line(color=v_color, dash="dash")))
    fig.add_trace(go.Scatter(
        x=ts, y=extruder_a, name="Linear Advance a",
        yaxis="y3",
        legendgroup="acceleration",
        line=go.scatter.Line(color=a_color, dash="dash")))
    fig.add_trace(go.Scatter(
        x=ts, y=extruder_j, name="Linear Advance j",
        yaxis="y4",
        legendgroup="jerk",
        line=go.scatter.Line(color=j_color, dash="dash")))
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


# %%
def graph_acceleration():
    a = 1000
    c = 0.05
    ts = np.linspace(0.0, 1.0, 100000, endpoint=True, dtype=np.float)
    toolhead_y = 0.5 * a * ts**2
    toolhead_v = a*ts
    toolhead_a = np.full(ts.shape[0], a)
    toolhead_j = np.full(ts.shape[0], 0)
    extruder_y = toolhead_y + c*toolhead_v
    extruder_v = a*ts + np.full(ts.shape[0],c*a)
    extruder_a = np.full(ts.shape[0], a)
    extruder_j = np.full(ts.shape[0], 0)
    graph_pressure(ts, toolhead_y, toolhead_v, toolhead_a, toolhead_j, extruder_y, extruder_v, extruder_a, extruder_j)
    
graph_acceleration()


# %%
def graph_smoothstep2():
    a = 1000
    c = 0.05
    ts = np.linspace(0.0, 1.0, 100000, endpoint=True, dtype=np.float)
    toolhead_y = 0.5*ts**4 + ts**3
    toolhead_v = -2*ts**3 + 3*ts**2
    toolhead_a = -6*ts**2 + 6 * ts
    toolhead_j = -12*ts + 6
    extruder_y = toolhead_y + c*toolhead_v
    extruder_v = toolhead_v + c*toolhead_a
    extruder_a = toolhead_a + c*toolhead_j
    extruder_j = toolhead_j - c*12
    graph_pressure(ts, toolhead_y, toolhead_v, toolhead_a, toolhead_j, extruder_y, extruder_v, extruder_a, extruder_j)
    
graph_smoothstep2()

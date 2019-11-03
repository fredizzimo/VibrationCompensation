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
import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.interpolate as interpolate
from ipycanvas import Canvas
from math import pi
import plotly.graph_objects as go

# %%
sym.init_printing()


# %%
def draw_system():
    canvas = Canvas(size=(500, 160))
    stepper_pos = (50, 10)
    stepper_size = (100, 150)
    stepper_pulley_pos = (
        stepper_pos[0] + stepper_size[0] / 2,
        stepper_pos[1] + 20)
    pulley_diameter = 20
    belt_gap = 3
    belt_top_y = stepper_pulley_pos[1] - pulley_diameter / 2 - belt_gap
    belt_bottom_y = stepper_pulley_pos[1] + pulley_diameter / 2 + belt_gap
    other_pulley_pos = (stepper_pulley_pos[0] + 300, stepper_pulley_pos[1])
    hotend_pos = (stepper_pulley_pos[0] + 200, belt_bottom_y)
    hotend_size = (20, 100)
    nozzle_size = (10, 15)
    hotend_middle_x = hotend_pos[0] + hotend_size[0] / 2
    hotend_top = hotend_pos[1] + hotend_size[1]
    spring_height = 10
    spring_advance = 10
    spring_pos = (stepper_pos[0] + stepper_size[0], belt_bottom_y + spring_height + 30)
    spring_length = hotend_pos[0] - spring_pos[0]
    damper_height = spring_height
    damper_pos = (spring_pos[0], spring_pos[1] + spring_height + damper_height + 5)
    damper_length = spring_length
    canvas.stroke_rect(*stepper_pos, *stepper_size)
    canvas.stroke_arc(*stepper_pulley_pos, pulley_diameter / 2, 0, 2*pi)
    canvas.stroke_arc(*other_pulley_pos, pulley_diameter / 2, 0, 2*pi)
    canvas.begin_path()
    canvas.move_to(stepper_pulley_pos[0], belt_top_y)
    canvas.line_to(other_pulley_pos[0], belt_top_y)
    canvas.arc(*other_pulley_pos, pulley_diameter / 2 + belt_gap, -pi/2, pi / 2, False)
    canvas.line_to(stepper_pulley_pos[0], belt_bottom_y)
    canvas.arc(*stepper_pulley_pos, pulley_diameter / 2 + belt_gap, pi / 2, pi + pi / 2, False)
    canvas.stroke()
    canvas.stroke_rect(*hotend_pos, *hotend_size)
    canvas.begin_path()
    canvas.move_to(hotend_middle_x - nozzle_size[0] / 2, hotend_top)
    canvas.line_to(hotend_middle_x, hotend_top + nozzle_size[1])
    canvas.line_to(hotend_middle_x + nozzle_size[0] / 2, hotend_top)
    canvas.stroke()
    def draw_spring(start, length):
        canvas.begin_path()
        canvas.move_to(*start)
        advance = spring_advance
        height = spring_height
        x = start[0] + advance / 2
        top = start[1] - height
        bottom = start[1] + height
        to = start[0] + length
        y_coord = [top, bottom]
        index = 0
        while x <= to:
            canvas.line_to(x, y_coord[index % 2])
            index += 1
            x += advance
        left = (to - x + advance) / advance
        if left > 0:
            if index % 2 == 1:
                end_y = bottom - height * left
            else:
                end_y = top + height * left
            canvas.line_to(start[0] + length, end_y)
        canvas.stroke()
    draw_spring(spring_pos, spring_length)
    def draw_damper(start, length):
        height = damper_height * 2
        canvas.begin_path()
        canvas.move_to(*start)
        end = (start[0] + length, start[1])
        middle = (start[0] + length / 2, start[1])
        y1 = start[1] - height / 2
        y2 = start[1] - height / 2 * 0.5
        y3 = start[1] + height / 2 * 0.5 
        y4 = start[1] + height / 2
        x1 = middle[0] - height / 2 * 0.5
        x2 = middle[0] + height / 2
        canvas.line_to(*middle)
        canvas.move_to(middle[0], y2)
        canvas.line_to(middle[0], y3)
        canvas.move_to(x1, y1)
        canvas.line_to(x2, y1)
        canvas.line_to(x2, y4)
        canvas.line_to(x1, y4)
        canvas.move_to(x2, middle[1])
        canvas.line_to(*end)
        canvas.stroke()
        
    draw_damper(damper_pos, damper_length)
    
    canvas.text_align = "center"
    canvas.fill_text("m", hotend_pos[0] + hotend_size[0] / 2, hotend_pos[1] + hotend_size[1] / 2)
    canvas.text_baseline = "bottom"
    canvas.fill_text("k", spring_pos[0] + spring_length / 2, spring_pos[1] - spring_height - 5)
    canvas.text_baseline = "top"
    canvas.fill_text("c", spring_pos[0] + spring_length / 2, damper_pos[1] + damper_height + 5)
    
    arrow_size = 5
    
    def draw_arrow(pos, length, text, position_arrow=False):
        canvas.begin_path()
        size = arrow_size
        if position_arrow:
            canvas.move_to(pos[0], pos[1] - size)
            canvas.line_to(pos[0], pos[1] + size)
        canvas.move_to(*pos)
        canvas.line_to(pos[0] + length, pos[1])
        canvas.move_to(pos[0] + length - size,pos[1] - size)
        canvas.line_to(pos[0] + length, pos[1])
        canvas.line_to(pos[0] + length - size, pos[1] + size)
        canvas.stroke()
        canvas.text_baseline = "top"
        canvas.fill_text(text, pos[0] + length / 2, pos[1] + size)
    
    draw_arrow((stepper_pulley_pos[0], stepper_pulley_pos[1] + pulley_diameter + 5), 30, "x", True)
    draw_arrow((hotend_middle_x, belt_top_y + 10), 30, "y", True)
    draw_arrow((hotend_pos[0] + hotend_size[0], hotend_pos[1] + hotend_size[1] / 2), 30, "F")
    
    def draw_wall(pos, height):
        step = 10
        width = 10
        y = pos[1]
        canvas.begin_path()
        canvas.move_to(pos[0], pos[1])
        while y <= height:
            y += step
            canvas.line_to(pos[0] - width, y - 2 * step)
            canvas.move_to(pos[0], y)
        canvas.stroke()
    
    draw_wall((stepper_pos[0], stepper_pos[1]), stepper_size[1])
    
    return canvas
draw_system()

# %%
s = sym.symbols("s")
f_g_i_s = sym.Function("G_i")(s)
f_x_i_s = sym.Function("X")(s)
f_y_i_s = sym.Function("Y")(s)
eq_g_i_s = sym.Eq(f_g_i_s, f_x_i_s / f_y_i_s)
display(eq_g_i_s)

# %%
x, y, k, c, m, t = sym.symbols("x, y, k, c, m, t")
f_y_t = sym.Function("y")(t)
f_x_t = sym.Function("x")(t)
eq_spring = sym.Eq((f_y_t-f_x_t)*k + c*(sym.Derivative(f_y_t, t) - sym.Derivative(f_x_t, t)), m * (sym.Derivative(f_y_t, (t,2))))
display(eq_spring)
eq_spring = sym.Eq(k*f_y_t + c*(sym.Derivative(f_y_t, t)) - m * (sym.Derivative(f_y_t, (t,2))), k*f_x_t + c*sym.Derivative(f_x_t, t))
display(eq_spring)

# %%
eq_spring_s = sym.Eq(k*f_y_i_s + c*s*f_y_i_s - m*s**2*f_y_i_s, k*f_x_i_s + c*s*f_x_i_s)
display(eq_spring_s)

# %%
eq_spring_g = sym.Eq(f_g_i_s, (eq_spring_s.lhs / f_y_i_s).simplify() / (eq_spring_s.rhs / f_x_i_s).simplify())
display(eq_spring_g)

# %%
eq_spring = sym.Eq(-(x)*k - c*(sym.Derivative(x, t)), m * (sym.Derivative(x, (t,2))))
display(eq_spring)
display(sym.laplace_transform(eq_spring.lhs - eq_spring.rhs, t, s))

# %%
s = sym.symbols("s")
w_n, z_n = sym.symbols("omega_n, zeta_n", positive=True)
f_g_i_s = sym.Function("G_i")(s)
eq_g_i_s = sym.Eq(f_g_i_s, (2*z_n*w_n*s + w_n**2) / (s**2 + 2*z_n*w_n*s + w_n**2))
display(eq_g_i_s)

# %%
g = sym.symbols("G")
f_a_s = sym.Function("a")(s)
f_b_s = sym.Function("b")(s)
eq_g = sym.Eq(g, f_b_s / f_a_s)
display(eq_g)

# %%
alpha_n = sym.IndexedBase("alpha")
beta_n = sym.IndexedBase("beta")
N, n = sym.symbols("N, n", integer=True, nonnegative=True)
eq_a_s = sym.Eq(f_a_s, sym.Sum(s**n * alpha_n[n], (n, 0, N)))
eq_b_s = eq_a_s.subs(f_a_s, f_b_s).subs(alpha_n, beta_n)
display(eq_a_s)
display(eq_b_s)

# %%
e = sym.symbols("e")
p, p_r, p_star_r = sym.symbols("p, p_r, {p_r}^*")
eq_e = sym.Eq(sym.Eq(e, p_r - p), p_r - eq_g.rhs * p_star_r, evaluate=False)
display(eq_e)

# %%
tau = sym.symbols("tau")
f_p_star_r = sym.Function("{p_r}^*")
f_p_star_r_t = f_p_star_r(t)
f_h = sym.Function("h")
f_p_r = sym.Function("p_r")
t0, t1 = sym.symbols("t_0, t_1")
eq_convolve_ref = sym.Eq(f_p_star_r_t, sym.Integral(f_h(tau)*f_p_r(t-tau), (tau, t0, t1)), evaluate=False)
display(eq_convolve_ref)

# %%
j = sym.symbols("j")
eq_constraints = sym.Eq(alpha_n[k] - sym.Sum(beta_n[k-j]*sym.Integral(f_h(tau)*tau**j, tau)*((-1)**j)/sym.factorial(j), (j, 0, k)))
display(eq_constraints)


# %%
def calculate_m():
    M = []
    for i in range(4):
        m_i = sym.symbols("M_%d" % i)
        integral = sym.Integral(f_h(tau)*tau**i, tau)
        substituted = eq_constraints.subs(k, i).doit()
        for j in range(i):
            substituted = substituted.subs(M[j].args[1].args[0], M[j].args[0])
        solved = sym.solve(substituted, integral)[0]
        eq = sym.Eq(m_i, sym.Eq(integral, solved))
        M.append(eq)
        display(eq)
    return M
eq_m_i = calculate_m()

# %%
eq_zrv_0 = sym.Eq(sym.Integral(f_h(t)*sym.exp(z_n*w_n*t)*sym.cos(t*w_n*sym.sqrt(1-z_n**2)), t))
eq_zrv_1 = eq_zrv_0.replace(sym.cos, sym.sin)
display(eq_zrv_0)
display(eq_zrv_1)

# %%
T = sym.symbols("T")
k_0, k_1 = sym.symbols("k_0, k_1")
eq_convolve_ref_disc = sym.Eq(f_p_star_r(m*T),T*sym.Sum(f_h(k*t)*f_p_r((m-k)*T), (k,k_0, k_1)))
display(eq_convolve_ref_disc)

# %%
f_m = sym.Function(r"\mathbf{m}")
d = sym.symbols(r"\mathbf{d}")
eq_constraints_disc = sym.Eq(sym.Sum(f_m(k)*f_h(k*T), (k, k_0, k_1)), d)
display(eq_constraints_disc)

# %%
eq_m_k = sym.Eq(f_m(k), sym.Matrix([
    1,
    k*T,
    k**2*T**2,
    k**3*T**3,
    sym.exp(z_n*w_n*k*T)*sym.cos(k*T*w_n*sym.sqrt(1-z_n**2)),
    sym.exp(z_n*w_n*k*T)*sym.cos(k*T*t*w_n*sym.sqrt(1-z_n**2))]),
    evaluate=False)
display(eq_m_k)

# %%
eq_d = sym.Eq(d, sym.Matrix([
    eq_m_i[0].args[0],
    eq_m_i[1].args[0],
    eq_m_i[2].args[0],
    eq_m_i[3].args[0],
    0,
    0
    ]), evaluate=False)
display(eq_d)

# %%
M,h = sym.symbols(r"\mathbf{M}, \mathbf{h}")
eq_constraints_matrix = sym.Eq(M*h, d)
display(eq_constraints_matrix)

# %%
eq_h = sym.Eq(h, )


# %%
def create_system(frequency, damping):
    w_0 = frequency * (2.0 * np.pi)
    z = damping
    return sp.signal.TransferFunction([2*w_0*z, w_0**2], [1, 2*w_0*z, w_0**2])


# %%
def plot_bode(system, n=1000):
    w, mag, phase = signal.bode(system, n)
    freq = w / (2.0 * np.pi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=mag))
    fig.update_layout(title="Bode Amplitude", xaxis_type="log")
    fig.update_xaxes(title="Frequency (Hz)")
    fig.update_yaxes(title="Amplitude (db)")
    fig.show()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq, y=phase))
    fig.update_layout(title="Bode Phase", xaxis_type="log")
    fig.update_xaxes(title="Frequency (Hz)")
    fig.update_yaxes(title="Phase (degrees)")
    fig.show()


# %%
def plot_impulse(system, n=1000):
    t, y= signal.impulse(system, N=n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y))
    fig.update_xaxes(title="Time (s)")
    fig.update_layout(title="Impulse response")
    fig.show()


# %%
def plot_step(system, n=1000):
    t, y= signal.step(system, N=n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y))
    fig.update_xaxes(title="Time (s)")
    fig.update_layout(title="Step response")
    fig.show()


# %%
system = create_system(30, 0.1)
plot_bode(system)
plot_impulse(system)
plot_step(system, 100000)


# %%
def plot_response(system, u, t, name=""):
    _, y, _= signal.lsim(system, u, t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=u, name="input"))
    fig.add_trace(go.Scatter(x=t, y=y, name="output"))
    fig.add_trace(go.Scatter(x=t, y=y-u, name="error"))
    fig.update_xaxes(title="Time (s)")
    fig.update_layout(title=name)
    fig.show()
    
def generate_curve(*points, num=1000):
    points = np.array([*points])
    interpolator = interpolate.interp1d(points[:,1], points[:,0])
    t = np.linspace(points[0,1], points[-1,1], num)
    v = interpolator(t)
    return v, t
    
plot_response(system, *generate_curve((0,0), (5, 0.1), (5, 1)), "Fixed speed, then stop")

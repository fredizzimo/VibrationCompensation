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
from ipycanvas import Canvas
from math import pi

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
s, w_n, z = sym.symbols("s, omega_n, zeta")
f_g_i_s = sym.Function("G_i")(s)
eq_g_i_s = sym.Eq(f_g_i_s, (w_n ** 2) / (s**2 + 2*z*w_n*s + w_n**2))
display(eq_g_i_s)

# %%
test_f = sym.Function("f")(t)
test = sym.Derivative(test_f, t)
display(test)
display(sym.laplace_transform(test, t, s))

# %%

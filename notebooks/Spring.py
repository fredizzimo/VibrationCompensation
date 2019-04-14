#%%
from IPython.display import display
import sympy as sp
sp.init_printing()

#%%
f_x_spring = sp.Function("x_spring")
f_x_toolhead = sp.Function("x_toolhead")
f_x_vibration = sp.Function("x_vibration")
s_t = sp.Symbol("t")
s_m = sp.Symbol("m")
s_k = sp.Symbol("k")
f_x = sp.Function("x")
eq_spring1 = sp.Eq(s_m*sp.Derivative(f_x_spring(s_t),s_t, 2), -s_k*(f_x_spring(s_t) - f_x_toolhead(s_t)))
eq_spring2 = sp.Eq(f_x_spring(0), 0)
eq_spring3 = sp.Eq(f_x_spring(s_t).diff(s_t).subs(s_t, 0), 0)

display(eq_spring1)
display(eq_spring2)
display(eq_spring3)
    
eq_spring4 = sp.dsolve(eq_spring1, ics={
    eq_spring2.lhs: eq_spring2.rhs,
    eq_spring3.lhs: eq_spring3.rhs,

})
display(eq_spring4)

eq_x_toolhead = sp.Eq(f_x_toolhead(s_t), sp.Piecewise((0, s_t <=0), (f_x(s_t), True)))
display(eq_x_toolhead)

eq_spring4 = eq_spring4.subs(f_x_toolhead(s_t), eq_x_toolhead.rhs).doit().simplify()
eq_spring4 = sp.Eq(eq_spring4.lhs, eq_spring4.rhs.args[1][0])
display(eq_spring4)

eq_vibration=sp.Eq(f_x_vibration(s_t), f_x_spring(s_t) - f_x(s_t)).simplify()
display(eq_vibration)

eq_vibration=eq_vibration.subs(f_x_spring(s_t), eq_spring4.rhs)
display(eq_vibration)
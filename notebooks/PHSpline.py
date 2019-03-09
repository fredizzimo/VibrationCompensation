#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'notebook'))
    print(os.getcwd())
except:
    pass

#%%
from IPython.display import display, Markdown, Math
import sympy as sp
sp.init_printing()


#%%
s_t = sp.Symbol("t")
s_i = sp.Symbol("i", integer=True)
s_z = sp.IndexedBase("z")
s_N = sp.Symbol("N", integer=True)
f_r = sp.IndexedBase(sp.Function("r")(s_t))
eq_27_1 = sp.Eq(
    sp.Derivative(f_r[s_i], s_t), 
        (
            sp.Rational(1, 2) * (s_z[s_i-1] + s_z[s_i]) * (1 - s_t)**2 +
            s_z[s_i] * 2 * (1 - s_t) * s_t +
            sp.Rational(1, 2) * (s_z[s_i] + s_z[s_i + 1]) * s_t**2
        )**2,
    evaluate = False
    )
display(eq_27_1)

#%%
def disabled():
    s_j = sp.Symbol("j", integer=True)
    r2 = sp.IndexedBase(sp.Function("r")(s_j / 4))
    eq_27_1_bernstein = sp.Sum(
        sp.binomial(4, s_j) * s_t**s_j * (1 - s_t)**(4 - s_j) * r2[s_i],
        (s_j, 0, 4))
    display(eq_27_1_bernstein)
    eq_27_1_bernstein = eq_27_1_bernstein.replace(r2[s_i], eq_27_1.rhs.subs(s_t, s_j / 4).simplify())
    display(eq_27_1_bernstein)
    display(eq_27_1_bernstein.doit().simplify().factor().expand(s_t))

#%%
dots = sp.Symbol("...")
f_f = sp.Function("f")(s_z[1], dots, s_z[s_N])
s_q = sp.IndexedBase("q")
delta_q = sp.IndexedBase(r"{\Delta}q")
eq_27_2 = sp.Eq(sp.Integral(eq_27_1.lhs, (s_t, 0, 1)), delta_q[s_i])
display(eq_27_2)
eq_27_2 = eq_27_2.replace(eq_27_1.lhs, eq_27_1.rhs).doit().simplify()
eq_27_2 = eq_27_2.rhs*60 - eq_27_2.lhs*60
eq_27_2 = eq_27_2.factor()
eq_27_2 = sp.Eq(f_f, eq_27_2)
display(eq_27_2)

#%%
s_p = sp.IndexedBase("p")
s_w = sp.IndexedBase("w")
eq_p0 = sp.Eq(s_p[s_i, 0], s_q[s_i-1])
eq_p1 = sp.Eq(s_p[s_i, 1], s_p[s_i, 0] + sp.Rational(1,5) * s_w[s_i,0]**2)
eq_p2 = sp.Eq(s_p[s_i, 2], s_p[s_i, 1] + sp.Rational(1,5) * s_w[s_i,0]*s_w[s_i,1])
eq_p3 = sp.Eq(s_p[s_i, 3], 
    s_p[s_i, 2] +
    sp.Rational(2,15) * s_w[s_i,1]**2 +
    sp.Rational(1, 15) * s_w[s_i,0]*s_w[s_i, 2])
eq_p4 = sp.Eq(s_p[s_i, 4], s_p[s_i, 3] + sp.Rational(1,5) * s_w[s_i,1]*s_w[s_i,2])
eq_p5 = sp.Eq(s_p[s_i, 5], s_p[s_i, 4] + sp.Rational(1,5) * s_w[s_i,2]**2)
display(eq_p0)
display(eq_p1)
display(eq_p2)
display(eq_p3)
display(eq_p4)
display(eq_p5)

#%%
eq_w0 = sp.Eq(s_w[s_i, 0], sp.Rational(1, 2) * (s_z[s_i-1] + s_z[s_i]))
eq_w1 = sp.Eq(s_w[s_i, 1], s_z[s_i])
eq_w2 = sp.Eq(s_w[s_i, 2], sp.Rational(1, 2) * (s_z[s_i] + s_z[s_i + 1]))
display(eq_w0)
display(eq_w1)
display(eq_w2)

#%%
s_k = sp.Symbol("k", integer=True)
eq_r_bezier = sp.Eq(
    f_r[s_i],
    sp.Sum(
            s_p[s_i,s_k]*sp.binomial(5, s_k) * 
            (1-s_t)**(5-s_k) * 
            s_t**s_k,
        (s_k, 0, 5)))
display(eq_r_bezier)

#%%
def calculate_line():
    # We have a line going from s_q[s_i-1] to s_q[s_i]
    # So let's fix the endpoints
    eq_line1 = sp.Eq(f_r[s_i].subs(s_t, 0), s_q[s_i-1])
    eq_line2 = sp.Eq(f_r[s_i].subs(s_t, 1), s_q[s_i])
    display(eq_line1)
    display(eq_line2)

    # Then the first derivative
    eq_line3 = sp.Eq(sp.Derivative(f_r[s_i], s_t).subs(s_t, 0), s_q[s_i] - s_q[s_i-1])
    eq_line4 = sp.Eq(sp.Derivative(f_r[s_i], s_t).subs(s_t, 1), s_q[s_i] - s_q[s_i-1])
    display(eq_line3)
    display(eq_line4)

    # Finally the second derivatives should be zero at the endpoints
    eq_line5 = sp.Eq(sp.Derivative(f_r[s_i], s_t, 2).subs(s_t, 0), 0)
    eq_line6 = sp.Eq(sp.Derivative(f_r[s_i], s_t, 2).subs(s_t, 1), 0)
    display(eq_line5)
    display(eq_line6)

    #display(eq_r_bezier.doit().diff(s_t, 2).subs(s_t, 0))
    expanded_curve = eq_r_bezier.rhs.doit()
    expanded_curve = expanded_curve.subs(eq_p5.lhs, eq_p5.rhs)
    expanded_curve = expanded_curve.subs(eq_p4.lhs, eq_p4.rhs)
    expanded_curve = expanded_curve.subs(eq_p3.lhs, eq_p3.rhs)
    expanded_curve = expanded_curve.subs(eq_p2.lhs, eq_p2.rhs)
    expanded_curve = expanded_curve.subs(eq_p1.lhs, eq_p1.rhs)
    expanded_curve = expanded_curve.subs(eq_p0.lhs, eq_p0.rhs)
    expanded_curve = expanded_curve.simplify()
    eq_expanded_r = sp.Eq(f_r[s_i], expanded_curve)
    eq_expanded_r_d1 = sp.Eq(sp.Derivative(eq_expanded_r.lhs, s_t), eq_expanded_r.rhs.diff(s_t)).simplify()
    eq_expanded_r_d2 = sp.Eq(sp.Derivative(eq_expanded_r_d1.lhs, s_t), eq_expanded_r_d1.rhs.diff(s_t)).simplify()

    display(eq_expanded_r)
    display(eq_expanded_r_d1)
    display(eq_expanded_r_d2)

    eq_line1 = sp.Eq(eq_expanded_r.rhs.subs(s_t, 0), eq_line1.rhs)
    eq_line2 = sp.Eq(eq_expanded_r.rhs.subs(s_t, 1), eq_line2.rhs)
    eq_line3 = sp.Eq(eq_expanded_r_d1.rhs.subs(s_t, 0), eq_line3.rhs)
    eq_line4 = sp.Eq(eq_expanded_r_d1.rhs.subs(s_t, 1), eq_line4.rhs)
    eq_line5 = sp.Eq(eq_expanded_r_d2.rhs.subs(s_t, 0), eq_line5.rhs)
    eq_line6 = sp.Eq(eq_expanded_r_d2.rhs.subs(s_t, 1), eq_line6.rhs)

    display("eq 1", eq_line1) # Always true
    display("eq 2", eq_line2) 
    display("eq 3", eq_line3) 
    display("eq 4", eq_line4)
    display("eq 5", eq_line5)
    display("eq 6", eq_line6)

    #If we arrange and simplify a bit
    eq_line7 = sp.Eq(s_w[s_i, 0]**2, sp.solve(eq_line5, s_w[s_i, 0]**2)[0])
    display("eq 7", eq_line7)

    eq_line8 = sp.Eq(s_w[s_i, 2]**2, sp.solve(eq_line6, s_w[s_i, 2]**2)[0])
    display("eq 8", eq_line8)

    #It's easy to see that 
    eq_line9 = sp.Eq(sp.Eq(s_w[s_i, 0], s_w[s_i, 1]), s_w[s_i, 2], evaluate=False)
    display("eq 9", eq_line9)

    #So let's use substitute
    eq_p0_line = eq_p0.subs(s_w[s_i, 1], s_w[s_i, 0]).subs(s_w[s_i, 2], s_w[s_i, 0])
    eq_p1_line = eq_p1.subs(s_w[s_i, 1], s_w[s_i, 0]).subs(s_w[s_i, 2], s_w[s_i, 0])
    eq_p2_line = eq_p2.subs(s_w[s_i, 1], s_w[s_i, 0]).subs(s_w[s_i, 2], s_w[s_i, 0])
    eq_p3_line = eq_p3.subs(s_w[s_i, 1], s_w[s_i, 0]).subs(s_w[s_i, 2], s_w[s_i, 0])
    eq_p4_line = eq_p4.subs(s_w[s_i, 1], s_w[s_i, 0]).subs(s_w[s_i, 2], s_w[s_i, 0])
    eq_p5_line = eq_p5.subs(s_w[s_i, 1], s_w[s_i, 0]).subs(s_w[s_i, 2], s_w[s_i, 0])

    display(eq_p0_line)
    display(eq_p1_line)
    display(eq_p2_line)
    display(eq_p3_line)
    display(eq_p4_line)
    display(eq_p5_line)

    # And the final form

    eq_p0_line = eq_p0_line.subs(eq_line3.lhs, eq_line3.rhs)
    eq_p1_line = eq_p1_line.subs(eq_line3.lhs, eq_line3.rhs)
    eq_p2_line = eq_p2_line.subs(eq_line3.lhs, eq_line3.rhs)
    eq_p3_line = eq_p3_line.subs(eq_line3.lhs, eq_line3.rhs)
    eq_p4_line = eq_p4_line.subs(eq_line3.lhs, eq_line3.rhs)
    eq_p5_line = eq_p5_line.subs(eq_line3.lhs, eq_line3.rhs)

    display(eq_p0_line)
    display(eq_p1_line)
    display(eq_p2_line)
    display(eq_p3_line)
    display(eq_p4_line)
    display(eq_p5_line)

    return eq_p0_line, eq_p1_line, eq_p3_line, eq_p4_line, eq_p5_line

eq_p0_line, eq_p1_line, eq_p3_line, eq_p4_line, eq_p5_line = calculate_line()
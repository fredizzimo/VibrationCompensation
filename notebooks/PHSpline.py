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

#%% [markdown]
# # The generic Bezier form

#%%
s_k = sp.Symbol("k", integer=True)
s_K = sp.Symbol("K", integer=True)
eq_r_bezier = sp.Eq(
    f_r[s_i],
    sp.Sum(
            s_p[s_i,s_k]*sp.binomial(s_K, s_k) * 
            (1-s_t)**(s_K-s_k) * 
            s_t**s_k,
        (s_k, 0, s_K)))
display(eq_r_bezier)

#%% [markdown]
# # Calculate the formula for a straight quintic line

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

    expanded_curve = eq_r_bezier.rhs.subs(s_K, 5).doit()
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

#%% [markdown]
# # The matrix form of ph splines

#%%
def generate_matrix_form(degree):
    # See https://pomax.github.io/bezierinfo/#matrix
    bezier_without_points = eq_r_bezier.rhs.replace(s_p[s_i,s_k], 1)
    expanded = bezier_without_points.subs(s_K, degree).doit()
    coeffs = [sp.Poly(arg).coeffs() for arg in expanded.args]
    coeffs = sorted(coeffs, key=len)
    coeffs = [coeff + [0]*(len(coeffs)-len(coeff)) for coeff in coeffs]
    coeff_matrix = sp.Matrix(coeffs)
    t_matrix = sp.Matrix([s_t**i for i in range(degree + 1)]).T
    p_matrix = sp.Matrix([s_p[s_i, i] for i in range(degree + 1)])
    return sp.MatMul(t_matrix, coeff_matrix, p_matrix)
display("Cubic", generate_matrix_form(3))
display("Quintic", generate_matrix_form(5))
display("Undecic (11th degree", generate_matrix_form(11))

#%% [markdown]
# # The splitting of ph splines

#%%
# See https://pomax.github.io/bezierinfo/#splitting
def generate_splitting_matrix_first(degree, splitpoint):
    matrix_form=generate_matrix_form(degree)
    _, M, P = matrix_form.args
    Z = sp.Matrix(degree+1, degree+1, lambda i,j: splitpoint**i if i==j else 0)
    Q = sp.MatMul(M.inverse(), Z, M).doit().simplify()
    return sp.MatMul(Q,P) 

def generate_splitting_matrix_second(degree, splitpoint):
    Q, P = generate_splitting_matrix_first(degree, splitpoint).args
    Q = Q.tolist()
    def rotate(l, n):
        return l[-n:] + l[:-n]
    Q = [rotate(r, len(r)-1-i) for i,r in enumerate(Q)]
    Q.reverse()
    Q = sp.Matrix(Q)
    return sp.MatMul(Q,P)

display(
    "Cubic", 
    generate_splitting_matrix_first(3, sp.Symbol("z")),
    generate_splitting_matrix_second(3, sp.Symbol("z"))
)
display(
    "Quintic", 
    generate_splitting_matrix_first(5, sp.Symbol("z")),
    generate_splitting_matrix_second(5, sp.Symbol("z"))
)

#%% [markdown]
# # The splititng matrices for the corner 11th degree PHSpline used by the CornerSmoother

#%%
def generate_corner_splitting_matrices():
    splitting_matrix_first_half = generate_splitting_matrix_first(11, sp.Rational(1,2)).doit()
    splitting_matrix_second_half = generate_splitting_matrix_second(11, sp.Rational(1,2)).doit()
    l = sp.Symbol("l")
    l_prime = sp.Symbol("l'")
    P1 = sp.Symbol("P1")
    T0 = sp.Symbol("T0")
    T1 = sp.Symbol("T1")
    T2 = sp.Symbol("T2")

    B0 = P1 - (3*l + l_prime) * T0
    B1 = P1 - (2*l + l_prime) * T0
    B2 = P1 - (l + l_prime) * T0
    B3 = P1 - l_prime * T0
    B4 = B3 + sp.Rational(5, 6) * l * T0 + sp.Rational(1, 6.0) * l * T2
    B5 = B4 + sp.Rational(10, 21) * l * T0 + sp.Rational(11, 21) * l * T2
    B6 = B5 + l * T2
    B7 = B6 + (sp.Rational(10, 21) * l) * T1 + sp.Rational(11, 21) * l * T2
    B8 = P1 + l_prime * T1
    B9 = P1 + (l + l_prime) * T1
    B10 = P1 + (2 * l + l_prime) * T1
    B11 = P1 + (3 * l + l_prime) * T1

    def substitute(m):
        return (m.
            subs(s_p[s_i,0], B0).
            subs(s_p[s_i,1], B1).
            subs(s_p[s_i,2], B2).
            subs(s_p[s_i,3], B3).
            subs(s_p[s_i,4], B4).
            subs(s_p[s_i,5], B5).
            subs(s_p[s_i,6], B6).
            subs(s_p[s_i,7], B7).
            subs(s_p[s_i,8], B8).
            subs(s_p[s_i,9], B9).
            subs(s_p[s_i,10], B10).
            subs(s_p[s_i,11], B11)
        ).simplify()
    return  substitute(splitting_matrix_first_half), substitute(splitting_matrix_second_half)

display(generate_corner_splitting_matrices())

#%% [markdown]
# # w paramater of a 11th degree PH spline

#%%
def calculate_w():
    # References 
    # [1] Hu, Qin, et al. "A Real-Time C3 Continuous Local Corner Smoothing and Interpolation Algorithm for CNC Machine Tools." Journal of Manufacturing Science and Engineering (2019): 1-37.
    # [2] Farouki, Rida T., Carlotta Giannelli, and Alessandra Sestini. "Identification and “reverse engineering” of Pythagorean-hodograph curves." Computer Aided Geometric Design 34 (2015): 21-36.
    s_w = sp.IndexedBase("w", complex=True)
    s_d = sp.IndexedBase("d", complex=True)
    w_fun = sp.Function("w")(s_t)
    d_fun = sp.Function("d")(s_t)
    s_b = sp.IndexedBase("b")
    def bernstein(variable, degree):
        return sp.Sum(
            variable[s_k]*sp.binomial(degree, s_k) * 
            (1-s_t)**(degree-s_k) * 
            s_t**s_k, (s_k, 0, degree))
    
    display("[2] Gives the following formula for reverse engineering a PH spline")
    eq_w = sp.Eq(w_fun, bernstein(s_w, (s_K-1) / 2))
    display(eq_w)
    eq_d = sp.Eq(d_fun, bernstein(s_d, s_K-1))
    display(eq_d)

    eq_w_d = sp.Eq(eq_w.rhs**2, eq_d.rhs)
    display(eq_w_d)

    display("We are dealing with a 11th degree spline")
    degree = 11
    eq_11_degree = eq_w_d.subs(s_K, degree)
    display(eq_11_degree)
    res = sp.solve(eq_11_degree.doit(), [s_d[i] for i in range(degree)], dict=True)
    eq_solved_d = [sp.Eq(k, v) for k, v in res[0].items()]
    for e in eq_solved_d:
        display(e)

    display("From [1] eq a7 we have")
    eq_a7_w = [
        sp.Eq(s_w[0], sp.Eq(s_w[1], s_w[2]), evaluate=False),
        sp.Eq(s_w[3], sp.Eq(s_w[4], s_w[5]), evaluate=False)
    ]
    display(eq_a7_w)

    display("Substituting with the above gives")

    for i,_ in enumerate(eq_solved_d):
        for a in eq_a7_w:
            eq_solved_d[i] = eq_solved_d[i].subs(a.rhs.lhs, a.lhs).subs(a.rhs.rhs, a.lhs)
    
    for e in eq_solved_d:
        display(e)

    display("We can use d0 and d5 to solve w0 and w3")
    # Note that we are ignoring the signs like [2] does
    w_0 = sp.Eq(s_w[0], sp.solve(eq_solved_d[0], s_w[0])[1])
    w_3 = sp.Eq(s_w[3], sp.solve(eq_solved_d[5].subs(w_0.lhs, w_0.rhs), s_w[3])[0])
    display(w_0)
    display(w_3)

    display("And from [2] we have")
    s_x = sp.IndexedBase("x")
    s_y = sp.IndexedBase("y")
    eq_d_p = sp.Eq(s_d[s_k], s_K*(s_p[s_k+1]-s_p[s_k]))
    eq_p_k = sp.Eq(s_p[s_k], s_x[s_k] + sp.I*s_y[s_k])
    eq_p_k1 = eq_p_k.subs(s_k, s_k+1)
    display(eq_d_p)
    display(eq_p_k)
    eq_d_p = eq_d_p.subs(eq_p_k.lhs, eq_p_k.rhs).subs(eq_p_k1.lhs, eq_p_k1.rhs)
    display(eq_d_p)

calculate_w()
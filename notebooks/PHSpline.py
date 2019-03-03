#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), 'notebook'))
    print(os.getcwd())
except:
    pass

#%%
from IPython.display import display
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
delta_q = sp.IndexedBase(r"{\Delta}q")
eq_27_2 = sp.Eq(sp.Integral(eq_27_1.lhs, (s_t, 0, 1)), delta_q[s_i])
display(eq_27_2)
eq_27_2 = eq_27_2.replace(eq_27_1.lhs, eq_27_1.rhs).doit().simplify()
eq_27_2 = eq_27_2.rhs*60 - eq_27_2.lhs*60
eq_27_2 = eq_27_2.factor()
eq_27_2 = sp.Eq(f_f, eq_27_2)
display(eq_27_2)

#%%
s_K = sp.Symbol("K", integer=True)
s_k = sp.Symbol("k", integer=True)
s_p = sp.IndexedBase("p")
eq_27_3_generic = sp.Eq(
    f_r,
    sp.Sum(
            s_p[s_i,s_k]*sp.binomial(s_K, s_k) * 
            (1-s_t)**(s_K-s_k) * 
            s_t**s_k,
        (s_k, 0, s_K)))
display(eq_27_3_generic)
eq_27_3_quintic = eq_27_3_generic.subs(s_K, 5)
display(eq_27_3_quintic)
eq_27_3_quintic = eq_27_3_quintic.doit()
display(eq_27_3_quintic)
test = sp.integrate(eq_27_1.rhs, (s_t, 0, sp.Rational(1, 5))).simplify()
w = sp.IndexedBase("w")

temp = 4
s_j = sp.Symbol("j", integer=True)
f_r2 = sp.IndexedBase(sp.Function("r")(s_j / temp))
test = sp.binomial(temp, s_j) * s_t**s_j * (1 - s_t)**(temp - s_j) * f_r2[s_i]
display(test)
#549
#257
test2 = eq_27_1.rhs.subs(s_t, sp.Rational(1, 4))
display(test2.simplify()*512)
#test = test.replace(r2[s_i], eq_27_1.rhs.subs(s_t, s_j / temp))
#display(test.subs(s_j, 1).simplify()*128)
#test.xreplace()

#%%
t0 = eq_27_1.rhs.subs(s_t, sp.Rational(0, 4).subs(s_i, 0).simplify())
t1 = eq_27_1.rhs.subs(s_t, sp.Rational(1, 4).simplify())
t2 = eq_27_1.rhs.subs(s_t, sp.Rational(2, 4).simplify())
t3 = eq_27_1.rhs.subs(s_t, sp.Rational(3, 4).simplify())
t4 = eq_27_1.rhs.subs(s_t, sp.Rational(4, 4).simplify())
display(t0)
display(t1)
display(t2)
display(t3)
display(t4)
display(eq_27_1)
display((sp.Rational(1,2)*(s_z[s_i-1] + s_z[s_i])*s_z[s_i]).simplify())
# this works, but not with deep=True
display(eq_27_1.rhs.expand(deep=False).collect([(1-s_t)**4], evaluate=False)[(1-s_t)**4])
display(eq_27_1.rhs.expand().simplify())
a = sp.Symbol("a")
temp = eq_27_1.rhs.expand().collect(s_t)
display(temp.collect((1-s_t)**4))
expanded = eq_27_1.rhs.expand(deep=False)
factors = [
        (1-s_t)**4,
        (1-s_t)**3 * s_t,
        (1-s_t)**2 * s_t**2,
        (1-s_t) * s_t**3,
        s_t**4
    ]
collected = expanded.collect(factors, evaluate=False)
for _k, _v in collected.items():
    print(_v)

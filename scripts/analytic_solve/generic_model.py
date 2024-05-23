# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:18:39 2023

@author: bwb1p
"""

"""
Having spent a bunch of time grizzling my way through the maths I'm going to 
attempt to create a more general solution. Starting out with a numerical model
like we've done before we can parameterise it with a simple polynomial fit.

Also, I'm going to try shifting it all to be exponentials instead of sines
and cosines. partly because I think it'll be easier for a general solution,
but also partly as an exercise.

This version is going to just create full symbolic outputs so we can load the 
equations in to a file later and sub in any C_x terms we want.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import sympy
from sympy import (
    expand,
    factor,
    Function,
    Rational,
    collect,
    factorial,
    simplify,
)
from sympy import (
    symbols,
    latex,
    multiline_latex,
    Poly,
    powsimp,
    cancel,
    lambdify,
)
from sympy import pi, sqrt, sympify, trigsimp

# import sympy as sp
from sympy import re, im, I, conjugate
from sympy.simplify.fu import TR8, TR10i
from tqdm import tqdm
import dill

# IMPORTANT! This determines how may polynomial orders (up to mu**(terms + 1)) we use
terms = 7

g = terms + 1  # We need a convenient term for loops later

# Model starts here

mu, sigma = symbols("mu sigma")

# For the exponential decay part, positions and times
x, t, x_0 = symbols("x t x_0", real=True)
omega_1, A_1, theta_1, phi_1, tau_1 = symbols(
    "omega_1 A_1 theta_1 phi_1 tau_1", real=True
)
omega_2, A_2, theta_2, phi_2, tau_2 = symbols(
    "omega_2 A_2 theta_2 phi_2 tau_2", real=True
)

# Set up the new symbols we'll need for a pair of decaying sinusoids at angular
# frequencies omega_1 and omega_2 with phase offsets omega_1 and omega_2
B_1, B_2 = symbols("B_1 B_2", real=True)  # Decaying amplitude
w_1, w_2 = symbols("w_1 w_2", real=True)
E_1, E_2 = symbols(
    "B_1 B_2", real=True
)  # The exponential bit of the decaying amplitude if needed

# This is the bit where we swap for complex exponentials Sine term versions, naturally.
mu_1 = Rational(0.5) * I * B_1 * (sympy.exp(I * w_1) - sympy.exp(-I * w_1))
mu_2 = Rational(0.5) * I * B_2 * (sympy.exp(I * w_2) - sympy.exp(-I * w_2))

print()
print(
    "Beam position mu expressed as a combination of sines (in exponential expansion form)"
)
print("mu :", mu_1 + mu_2 + x_0)
# print('mu :',latex(mu_1 + mu_2 + x_0))

# Equation and dictionary of the symbols we'll need to represent the C_x coefficients
PD_poly = symbols("PD_poly")
C_x = symbols("C_:%d" % g, real=True)

PD_poly = 0

# Create the basic function
for p in range(0, g):
    PD_poly = PD_poly + C_x[p] * mu**p

print()
print("Generic function into which we insert our equations")
print("PD_poly = ", PD_poly)
print()

# Now create a dictionary containing the coefficients themselves
a = expand(PD_poly)
mu_coef = {mu**p: a.coeff(mu, p) for p in range(0, g)}

print("List of C_x coefficients for mu**x up to x = ", terms, "terms")
for v in range(0, len(C_x)):
    print(C_x[v], " = ", mu_coef[mu**v])
print()

# Now setting up some symbols where we'll need to keep a running total.
b, b_p = symbols("b b_p")
b = 0

# Here we expand out the mu**x terms and construct the basic equation in terms
# of B_1, B_2 and C_x.
print("Exctracting mu**p terms in P_diff")
print()

for p in range(0, g):
    if (
        mu_coef[mu**p] != 0
    ):  # Given a power of mu (mu**p) where the coefficient exists
        b_p1 = (mu**p).subs(
            mu, mu_1 + mu_2 + x_0
        )  # create an expression with the mu**p with actual position subbed in.
        b_p = expand(b_p1)  # Then expand out the polynomial
    else:
        b_p = 0  # Or we set it to zero if the coefficient doesn't exist.

    b = b + C_x[p] * b_p  # Sum of all individual mu**p components
    print(
        C_x[p], ": Expansion complete"
    )  # Simple indicator of how far through the expansion we are

d_1 = expand(simplify(expand(b)))
"""
# Output the individual terms if necessary
print()
print('All the expanded mu**x terms summed together')
print(d_1)
print()
# print(latex(d_1).replace('} + \\frac{','} \\\\ \n & + \\frac{').replace('} - \\frac{','} \\\\ \n & - \\frac{'))
for v in d_1.args:
    print(v)
# print(d_1)
print()

# Now we extract only the terms at the measurment peak frequency
w_p = symbols('w_p')
d_2 = symbols('d_2')
d_2 = 0

for v in d_1.args:
    v1 = v.subs([(w_1,w_p),(w_2,w_p)])
    if v1.coeff(sympy.exp(I*w_p), 1) != 0:
        d_2 = d_2 + v

print('Just w_1 and w_2 terms')
for v in d_2.args:
    print(v)
print()

# Right! This is the signal that goes into the DAC. No substitutions 
# this time. We'll keep all of those for the end.

"""

# Still missing a 2w term in the final output. Trying a quadrature demodulation instead of
# a purely in-phase approach.

w_p = symbols("w_p")

# d_3_I = expand(d_1 * I*Rational(0.5)*(sympy.exp(I*w_1) - sympy.exp(-I*w_1)))    # Sine
# d_3_Q = expand(d_1 *   Rational(0.5)*(sympy.exp(I*w_1) + sympy.exp(-I*w_1)))    # Cosine

d_3_I = expand(
    d_1
    * I
    * Rational(0.5)
    * (sympy.exp(I * (w_1 + w_2) / 2) - sympy.exp(-I * (w_1 + w_2) / 2))
)  # Sine
d_3_Q = expand(
    d_1
    * Rational(0.5)
    * (sympy.exp(I * (w_1 + w_2) / 2) + sympy.exp(-I * (w_1 + w_2) / 2))
)  # Cosine

# Apply Low Pass Filtering
d_3_I_LPF = d_3_I
d_3_Q_LPF = d_3_Q

print("Expanding In-phase demod terms:")
for n, v in tqdm(enumerate(d_3_I.args), total=len(d_3_I.args)):
    v1 = v.subs([(w_1, w_p), (w_2, w_p)])
    for p in range(1, 2 * g):
        if v1.coeff(sympy.exp(I * w_p), p) != 0:
            d_3_I_LPF -= v

print()
print("Expanded In-phase demod terms")
print(d_3_I_LPF)
print()

print("Expanding Quad-phase demod terms:")
for n, v in tqdm(enumerate(d_3_Q.args), total=len(d_3_Q.args)):
    v1 = v.subs([(w_1, w_p), (w_2, w_p)])
    for p in range(1, 2 * g):
        if v1.coeff(sympy.exp(I * w_p), p) != 0:
            d_3_Q_LPF -= v

print()
print("Expanded Quad-phase demod terms")
print(d_3_Q_LPF)
print()

# d_3_I_trig = d_3_I.rewrite(sympy.cos).expand().as_real_imag()
# d_3_Q_trig = d_3_Q.rewrite(sympy.cos, sympy.sin).expand().as_real_imag()


# The voltage at the measurment frequency gets squared up (complex number times conjugate)
# Then filtered to get only the terms inside the measurement bin
print("Here begins the squaring up and filtering part")

# # func_1 = d_2*conjugate(d_2)
# func_1 = d_1*conjugate(d_1)
func_1 = d_3_I_LPF * conjugate(d_3_I_LPF) + d_3_Q_LPF * conjugate(d_3_Q_LPF)

#'''
# print('Basic equation')
# print(func_1)
# print()
# print('Basic equation - LaTeX version')
# print(latex(func_1))
# print()
# '''

# Expand it out to get something we can work with
func_2 = symbols("func_2")
func_2 = expand(func_1)

# '''
print("Basic equation - squared")
# print(func_2)
print("Complete")
print()
# print('Basic equation squared - LaTeX version')
# print(latex(func_2))
# print()
# '''

#  Now we filter it
DC_terms, AC_terms = symbols("DC_terms AC_terms")
AC_terms2, AC_terms3 = symbols("AC_terms2 AC_terms3")
All_terms = symbols("All_terms")

All_terms = []
for p in range(0, 2 * g):
    All_terms.append(0)

DC_terms = 0
AC_terms = 0

# by running through the terms and keeping only existing powers of exp(I*w_1)*exp(-I*w_2)
print("Filtering AC and DC terms:")
for n, v in tqdm(enumerate(func_2.args), total=len(func_2.args)):
    # print(str(v))
    for p in range(-2 * g, 2 * g):
        if p != 0:
            if v.coeff(sympy.exp(I * w_1) * sympy.exp(-I * w_2), p) != 0:
                AC_terms = AC_terms + v
    if "exp" not in str(v):
        DC_terms = DC_terms + v

print()
print("DC_terms:")
print(simplify(DC_terms))
print()
# print('AC_terms:')
# print(AC_terms)
# print()

# Convert exponentials to trigonometric form - essentially all cosines
AC_terms2 = AC_terms.rewrite(sympy.cos).expand().as_real_imag()
AC_terms3 = 0

print("AC terms - exponentials converted to trig form")
for v in AC_terms2:
    for v1 in v.args:
        v2 = expand(TR8(v1))
        #        print('X:\t', v2)
        AC_terms3 += v2

AC_terms = AC_terms3

for p in range(1, g * 2):
    AC_terms = AC_terms.collect(sympy.cos(p * w_1 - p * w_2))

print()
print("AC terms - reduced to only terms that fall within the measurement band")
# print(AC_terms)
for v in AC_terms.args:
    print("X:\t", v)

# # DC_terms = DC_terms / 1.01
func_4 = simplify(DC_terms) + AC_terms

print()
print("Final term simplified")
# print(func_4)
print("f(t)^2_{meas,terms=%d} = &" % terms, latex(func_4))
print()

# # Printing out the separate terms so we can get a clearer look at how it's all put tohether
# # Might be possible to simplify the terms a bit.

func_5 = symbols("func_5")
func_5 = 0

print("Final term in separate multiples of beat frequency")
All_terms[0] = DC_terms
for p in range(1, g):
    All_terms[p] = AC_terms.coeff(sympy.cos(p * w_1 - p * w_2)) * sympy.cos(
        p * w_1 - p * w_2
    )

for p in range(0, g):
    All_terms[p] = All_terms[p].collect(B_1)
    All_terms[p] = All_terms[p].collect(B_2)

for p in range(0, g):
    # print('%ddw term =' % p, All_terms[p])
    print("%ddw term =" % p, latex(All_terms[p]))
    print()
    func_5 += All_terms[p]

# print()
# print('Final term simplified further')
# # print(func_5)
# print('f(t)^2_{meas,terms=%d} = &' % terms,latex(func_5))
# print()

# # Now we take this equation and try to evaluate it for comparison to the
# # decaying sinusoid data.
# # First we substitute in the actual terms for C_x, B)x and w_x

func_6 = func_5  # Use the simplified version
# func_6 = func_4

# Then sub in the exponential decays and the actual frequencies and phases.
#  Leave out if you want to do something a bit more complicated later.
# func_6 = func_6.subs(B_1, A_1 * sympy.exp(-t/tau_1))
# func_6 = func_6.subs(B_2, A_2 * sympy.exp(-t/tau_2))
# func_6 = func_6.subs(w_1, omega_1 * t + phi_1)
# func_6 = func_6.subs(w_2, omega_2 * t + phi_2)

print("Final term")
print(func_6)
print()

# Export equation to file
filename = str("General_Equation_%d_Terms_Symbolic.txt" % terms)
filename2 = str("General_Equation_%d_Terms_Object.txt" % terms)
print("Saving equation as :", filename)
print("And as :", filename2)

with open(filename, "w") as text_file:
    text_file.write(str(func_6))

dill.settings["recurse"] = True
dill.dump(func_6, open(filename2, "wb"))

# Convert the symbolic function to a callable numpy version
# Again, leave out if doing more complicated stuff in another file later.
# fsqr_total = lambdify([t, A_1, omega_1, phi_1, tau_1, A_2, omega_2, phi_2, tau_2, x_0], func_6)

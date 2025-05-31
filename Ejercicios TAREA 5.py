# =============================================================================
# PROBLEMA 3 SET 7.1
# =============================================================================
"""
Integrar:
    y=siny   ,   y(0) = 1
desde x=0 hasta x=0.5 con el método de Euler usando h = 0.1
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sin, tan, atan, exp

#Condiciones iniciales
x0 = 0.0
y0 = 1.0
xStop = 0.5
h = 0.1

#dy/dx = sin(y)
def F(x, y):
    return sin(y)

#Solución exacta
def y_exacta(x):
    return 2 * atan(tan(0.5) * exp(x))

#Método de Euler
def eulerint(F, x, y, xStop, h):
    X = []
    Y = []
    X.append(x)
    Y.append(y)
    while x < xStop:
        h = min(h, xStop - x)
        y = y + h * F(x, y)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

X, Y = eulerint(F, x0, y0, xStop, h)

#Imprimir tabla errores
print("\n   x       y_Euler   y_Exacta   Error Absoluto")
print("------------------------------------------------")
for i in range(len(X)):
    ye = y_exacta(X[i])
    err = abs(Y[i] - ye)
    print(f"{X[i]:.2f}    {Y[i]:.6f}   {ye:.6f}   {err:.6f}")

#Graficar
x_vals = np.linspace(0, 0.5, 200)
y_vals = [y_exacta(x) for x in x_vals]

plt.plot(x_vals, y_vals, label="Solución exacta")
plt.plot(X, Y, "o--", label="Euler")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()

#%%
# =============================================================================
# PROBLEMA 4 SET 7.1
# =============================================================================
"""
Verificar que el problema:
    y'=y^(1/3)   ,   y(0) = 0
tiene dos soluciones:
    y=0 y y=(2x/3)^(3/2)
¿Cuál de las soluciones se reproduciría mediante integración numérica si la
condición inicial se establece en (a) y=0 y (b) y=10^-16?
"""
import matplotlib.pyplot as plt
import numpy as np

# Parámetros
x0 = 0.0
xStop = 1.0
h = 0.01

#Runge-Kutta 4
def Run_Kut4(F, x, y, xStop, h):
    def run_kut4(F, x, y, h):
        K0 = h * F(x, y)
        K1 = h * F(x + h / 2.0, y + K0 / 2.0)
        K2 = h * F(x + h / 2.0, y + K1 / 2.0)
        K3 = h * F(x + h, y + K2)
        return (K0 + 2.0 * K1 + 2.0 * K2 + K3) / 6.0

    X = [x]
    Y = [y]
    while x < xStop:
        h = min(h, xStop - x)
        y = y + run_kut4(F, x, y, h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

#dy/dx = y^{1/3}
def F(x, y):
    return y**(1/3)

#Solución exacta
def y_exacta(x):
    return ((2 * x) / 3)**(3/2)

#Solución numérica
X0, Y0 = Run_Kut4(F, x0, 0.0, xStop, h)
X1, Y1 = Run_Kut4(F, x0, 1e-16, xStop, h)

#Solución exacta
Y_exact = y_exacta(X1)

#Imprimir tabla valores
print("\nCaso (a): y(0) = 0")
print("  x       y_numérica      y_exacta     error_relativo (%)")
print("------------------------------------------------------------")
for i in range(0, len(X0), 10):
    err = 0.0  # porque todo es 0
    print(f"{X0[i]:.2f}    {Y0[i]:.6e}    {0.0: .6e}     {err: .6f}")

print("\nCaso (b): y(0) = 10^-16")
print("  x       y_numérica      y_exacta     error_relativo (%)")
print("------------------------------------------------------------")
for i in range(0, len(X1), 10):
    exact = y_exacta(X1[i])
    err = abs((Y1[i] - exact) / exact) * 100 if exact != 0 else 0.0
    print(f"{X1[i]:.2f}    {Y1[i]:.6e}    {exact:.6e}     {err: .6f}")

#Graficar
X_exact = np.linspace(0, xStop, 200)
Y_exact_plot = y_exacta(X_exact)

plt.plot(X0, Y0, label="y(0) = 0", linestyle='--')
plt.plot(X1, Y1, label="y(0) = 10^-16", linestyle='-.')
plt.plot(X_exact, Y_exact_plot, label="Solución exacta", color="black")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.show()
#%%
# =============================================================================
# PROBLEMA 3 SET 8.1
# =============================================================================
"""
Plantea aproximadamente la solución de los siguientes problemas de valor de
frontera. Usa el planteamiento para estimar y'(0) en cada problema:
    a) y"=-e^(-y) con y(0)=1 y y(1)=0.5

    c) y"=cos(xy) con y(0)=0 y y(1)=2
"""
import numpy as np
import matplotlib.pyplot as plt

#PARÁMETROS
x = 0.0
xStop = 1.0
h = 0.1

#Runge-Kutta 4
def Run_Kut4(F, x, y, xStop, h):
    def run_kut4(F, x, y, h):
        K0 = h * F(x, y)
        K1 = h * F(x + h / 2.0, y + K0 / 2.0)
        K2 = h * F(x + h / 2.0, y + K1 / 2.0)
        K3 = h * F(x + h, y + K2)
        return (K0 + 2.0 * K1 + 2.0 * K2 + K3) / 6.0
    X = [x]
    Y = [y]
    while x < xStop:
        h = min(h, xStop - x)
        y = y + run_kut4(F, x, y, h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

#Imprimir los resultados
def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n x ", end=" ")
        for i in range(n):
            print(f" y[{i}] ", end=" ")
        print()
    def imprimeLinea(x, y, n):
        print(f"{x:13.4e}", end=" ")
        for i in range(n):
            print(f"{y[i]:13.4e}", end=" ")
        print()
    m = len(Y)
    try:
        n = len(Y[0])
    except TypeError:
        n = 1
    if frec == 0:
        frec = m
    imprimeEncabezado(n)
    for i in range(0, m, frec):
        imprimeLinea(X[i], Y[i], n)
    if i != m - 1:
        imprimeLinea(X[m - 1], Y[m - 1], n)

#Método de Ridder
def Ridder(f, a, b, tol=1.0e-9):
    fa = f(a)
    if fa == 0.0:
        return a
    fb = f(b)
    if fb == 0.0:
        return b
    if np.sign(fa) != np.sign(fb):
        c = a
        fc = fa
    for i in range(30):
        c = 0.5 * (a + b)
        fc = f(c)
        s = np.sqrt(fc ** 2 - fa * fb)
        if s == 0.0:
            return None
        dx = (c - a) * fc / s
        if (fa - fb) < 0.0:
            dx = -dx
        x = c + dx
        fx = f(x)
        if i > 0:
            if abs(x - xOld) < tol * max(abs(x), 1.0):
                return x
        xOld = x
        if np.sign(fc) == np.sign(fx):
            if np.sign(fa) != np.sign(fx):
                b = x
                fb = fx
            else:
                a = x
                fa = fx
        else:
            a = c
            b = x
            fa = fc
            fb = fx
    print("Demasiadas iteraciones")
    return None

#Inciso a)
#y"=-exp(-y), y(0)=1, y(1)=0.5
def F_a(x, y):
    F = np.zeros(2)
    F[0] = y[1]
    F[1] = -np.exp(-y[0])
    return F

def initCond_a(u):
    return np.array([1.0, u])

def r_a(u):
    X, Y = Run_Kut4(F_a, x, initCond_a(u), xStop, h)
    y = Y[-1]
    return y[0] - 0.5

#Inciso c)
#y"=cos(x*y), y(0)=0, y(1)=2
def F_c(x, y):
    F = np.zeros(2)
    F[0] = y[1]
    F[1] = np.cos(x * y[0])
    return F

def initCond_c(u):
    return np.array([0.0, u])

def r_c(u):
    X, Y = Run_Kut4(F_c, x, initCond_c(u), xStop, h)
    y = Y[-1]
    return y[0] - 2.0

#SOLUCIÓN a)
u_a = Ridder(r_a, -5.0, 5.0)
X_a, Y_a = Run_Kut4(F_a, x, initCond_a(u_a), xStop, h)
print("Inciso (a): y'' = -e^-y")
print(f"Estimación de y'(0): {u_a:.6f}")
imprimeSol(X_a, Y_a, 2)

#SOLUCIÓN c)
u_c = Ridder(r_c, 0.0, 10.0)
X_c, Y_c = Run_Kut4(F_c, x, initCond_c(u_c), xStop, h)
print("\nInciso (c): y'' = cos(xy)")
print(f"Estimación de y'(0): {u_c:.6f}")
imprimeSol(X_c, Y_c, 2)

#Graficar
plt.plot(X_a, Y_a[:, 0], 'o-', label="Inciso (a): y'' = -e^-y")
plt.plot(X_c, Y_c[:, 0], 's-', label="Problema (c): y'' = cos(xy)")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.grid(True)
plt.legend()
plt.show()

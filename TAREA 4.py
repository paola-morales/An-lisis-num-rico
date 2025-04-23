# =============================================================================
# EJERCICIO 4.1 - 11
# =============================================================================
print('Ejercicio 1')
"""
Calcular todas las raíces de f(x) = 0 en un intervalo dado utilizando el 
método de Newton-Raphson. Probar el programa encontrando las raíces de la
ecuación x*sinx + 3cosx - x = 0 en (-6, 6).
"""
import numpy as np
import matplotlib.pyplot as plt

def err(string):
  print(string)
  input('Press return to exit')
  sys.exit()

def newtonRaphson(f,df,a,b,tol=1.0e-9):
  from numpy import sign
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa) == sign(fb): err('La raiz no esta en el intervalo')
  x = 0.5*(a + b)
  for i in range(30):
    print(i)
    fx = f(x)
    if fx == 0.0: return x 
    if sign(fa) != sign(fx): b = x # Haz el intervalo mas pequeño
    else: a = x
    dfx = df(x)  
    try: dx = -fx/dfx # Trata un paso con la expresion de Delta x
    except ZeroDivisionError: dx = b - a # Si division diverge, intervalo afuera
    x = x + dx # avanza en x
    if (b - x)*(x - a) < 0.0: # Si el resultado esta fuera, usa biseccion
      dx = 0.5*(b - a)
      x = a + dx 
    if abs(dx) < tol*max(abs(b),1.0): return x # Checa la convergencia y sal
  print('Too many iterations in Newton-Raphson')
def f(x): return x* np.sin(x) + 3* np.cos(x) - x
def df(x): return x*np.cos(x) - 2 *np.sin(x) - 1
root = newtonRaphson(f,df,-6,6)
print('Root =',root)


ax = plt.subplot(111)

t = np.arange(-7, 7, 0.25)
s = t* np.sin(t) + 3* np.cos(t) - t
line, = plt.plot(t, s, lw=2)

plt.annotate('raiz', xy=(2.0,0.0), xytext=(3.0, 20.0),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.grid(True)
plt.ylim(-40.0, 60.0)

plt.title("y = x* np.sin(x) + 3* np.cos(x) - x")
plt.ylabel('y = f(x)')
plt.xlabel('x')
plt.show()
#%%
# =============================================================================
# EJERCICIO 4.1 - 19
# =============================================================================
print('Ejercicio 2')
"""
La velocidad v de un cohete Saturno V en vuelo vertical cerca de la superficie
de la Tierra se puede aproximar con la ecuación:
    v = u*ln[(M_0)/(M_0 - mt)] - gt
  donde:
      u = 2,510 m/s = velocidad de escape relativo del cohete.
      M_0 = 2.8×10^6 kg = masa del cohete al momento del despegue.
      m = 13.3×10^3 kg/s = tasa de consumo del combustible.
      g = 9.81 m/s^2 = aceleración gravitacional.
      t = tiempo medido desde el despegue.
Determinar el tiempo en el que el cohete alcanza la velocidad del sonido
(335 m/s).
"""
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import sign
import sys
def ridder(f,a,b,tol=1.0e-9):
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa)!= sign(fb): c=a; fc=fa
  for i in range(30):
# Compute the improved root x from Ridder’s formula
      c = 0.5*(a + b); 
      fc = f(c)
      s = math.sqrt(fc**2 - fa*fb)
      if s == 0.0: return None
      dx = (c - a)*fc/s
      if (fa - fb) < 0.0: dx = -dx
      x = c + dx; fx = f(x)
# Test for convergence
  if i > 0:
     xOld = x
     if abs(x - xOld) < tol*max(abs(x),1.0): return x
# Re-bracket the root as tightly as possible
  if sign(fc) == sign(fx):
    if sign(fa)!= sign(fx): b = x; fb = fx
    else: a = x; fa = fx
  else:
    a = c; b = x; fa = fc; fb = fx
  return None
  print('Too many iterations')
  
u = 2510 #m/s
M_0 = 2.8e6 #kg
m = 13.3e3 #kg/s
g = 9.81 #m/s^2
v = 335 #m/s  

ax = plt.subplot(111)
t = np.arange(62, 75, 0.25)
s = u * np.log(M_0 / (M_0 - m * t)) - g * t - v
line, = plt.plot(t, s, lw=2)

plt.annotate('raiz', xy=(50.0,100.0), xytext=(1.5, 5.0),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.grid(True)
plt.ylim(-50, 50)

plt.title("y = u*ln[(M_0)/(M_0 - mt)] - gt - v")
plt.ylabel('y = f(x)')
plt.xlabel('x')
plt.show()


def f(t):
  return u * np.log(M_0 / (M_0 - m*t)) - g*t - v
a = 70
b = 72
t_resultado = ridder(f,a,b)
print('El tiempo que tarda en cohete en alcanzar la velocidad del sonido son', '{:6.4f}'.format(t_resultado),'segundos.')
print()
#%%
# =============================================================================
# EJERCICIO 5.1 - 9
# =============================================================================
print('Ejercicio 3')
"""
Usa los datos de la tabla para calcular:
    𝑓′(0.2)
lo más precisamente posible.

  x |    0   |  0.1   |  0.2   |  0.3   |  0.4   |
𝑓(x)|0.000000|0.078348|0.138910|0.192916|0.244981|
"""
x = [0.0, 0.1, 0.2, 0.3, 0.4]
fx = [0.000000, 0.078348, 0.138910, 0.192916, 0.244981]

"""
Utilizamos la siguiente estructura para encntrar la primera derivada:
    Aprox. con diferencias finitas: Primeras Centrales
    f'(x) = [f(x + h) - f(x - h)] / 2h
"""
h = 0.1 #ya que la diferencia entre nuestras x es de 0.1
#(x + h) = 0.2 + 0.1 = 0.3, que está en la posición 3
#(x - h) = 0.2 - 0.1 = 0.1, qu está en la posición 1
dfx = (fx[3] - fx[1]) / (2 * h)
print("f'(0.2) ≈", dfx)
print()
#%%
# =============================================================================
# EJERCICIO 5.1 - 10
# =============================================================================
print('Ejercicio 4')
"""
Determinar:
    𝑑(sin𝑥)/𝑑𝑥
en 𝑥 = 0.8 usando (a) la primera aproximación por diferencias hacia adelante
y (b) la primera aproximación por diferencias centradas. En cada caso, usa un
ℎ que dé el resultado más preciso (esto requiere experimentación).
"""
from sympy import symbols, diff, sin, lambdify
x = symbols('x')
f = sin(x)
df = diff(f, x)
f = lambdify(x, df)

print("La derivada de sen(x) en x=0.8 es: ", round(f(0.8),6),"\n")

def f(x,n): #La función a derivar con n decimales
  return round(sin(x),6)

#(a) Primera derivada de f con aproximación forward con n decimales
def dff(x,h,f,n): 
  dff = (4*f(x+h,n) - f(x+ 2*h,n) - 3*f(x,n)) / (2*h)
  return dff
print(round(dff(0.8, 0.019, f, 6), 6),"con h = 0.019\n")

#(b) Primera derivada de f con aproximación central con n decimales
def dfc(x,h,f,n): 
  dfc=(f(x+h,n) + f(x-h,n)) / (2 * h)
  return dfc
print(round(dfc(0.8, 0.7519, f, 6), 6),"con h = 0.7519")
print()
#%%
# =============================================================================
# EJERCICIO 6.1 - 1
# =============================================================================
print('Ejercicio 5')

"""
Usar la regla trapezoidal recursiva para evaluar:
    ∫ln(1+tanx)dx
    Con limites desde 0 hasta π/4
"""
def trapecio_recursiva(f,a,b,Iold,k):
  if k == 1: Inew = (f(a) + f(b))*(b - a)/2.0
  else:
    n = 2**(k -2 ) # numero de nuevos puntos
    h = (b - a)/n # espaciamiento de nuevos puntos
    x = a + h/2.0
    sum = 0.0
    for i in range(n):
      sum = sum + f(x)
      x = x + h
      Inew = (Iold + h*sum)/2.0
  return Inew

import math
def f(x):
    return math.log(1 + math.tan(x))
Iold = 0.0
for k in range(1,21):
  Inew = trapecio_recursiva(f,0.0,math.pi/4,Iold,k)
  if (k > 1) and (abs(Inew - Iold)) < 1.0e-6: break
  Iold = Inew

print('Integral =',round(Inew, 6))
print('n Panels =',2**(k-1))

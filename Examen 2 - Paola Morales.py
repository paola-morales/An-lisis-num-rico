'EXAMEN 2'
# =============================================================================
# Ejercicio 1
# =============================================================================
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
  
def f(x): return x**3 - 75 #Debemos hacer una funión
def df(x): return 3 * x**2
root = newtonRaphson(f,df,-50,50)
print('resultado =',round(root,3))

import matplotlib.pyplot as plt
import numpy as np

# Graficar funciones
t = np.arange(-50.0, 50.0, 0.01)
s = t**3 - 75  
line, = plt.plot(t, s, lw=2)


plt.grid(True)
plt.ylim(-50, 50)
plt.title("$y=x^3 - 75$")
plt.ylabel('$y = f(x)$')
plt.xlabel('x')
plt.show()

#%%
# =============================================================================
# Ejercicio 2
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(-4, 5, 0.01)
s = t**3 - 3.23*t**2 - 5.54*t + 9.84
line, = plt.plot(t, s, lw=2)

plt.grid(True)
plt.ylim(-15, 15)
plt.title("$y=x^3 - 3.23x^2 - 5.54x + 9.84$")
plt.ylabel('$y = f(x)$')
plt.xlabel('x')
plt.show()

def y(x):                    # define la funcion y(x)
  y = x**3 - 3.23*x**2 - 5.54*x + 9.84
  return y

x1 = 1 
x2 = 2 
y1 = y(x1)                                    # evalua la funcion y(x1)
y2 = y(x2)                                    # evalua la funcion y(x1)

if y1*y2 > 0:                                 # prueba si los signos son iguales
  print('No hay raices en el intervalo')
  exit

for i in range(100):
  xh = (x1+x2)/2
  yh = y(xh)                                  # evalua la funcion y(xh)
  y1 = y(x1)                                  # evalua la funcion y(x1)
  if abs(y1) < 1.0e-6:
    break
  elif y1*yh < 0:
    x2 = xh
  else:
    x1 = xh
print('La raiz es: %.5f' % x1)
print('Numero de bisecciones: %d' % (i+1))
#%%
# =============================================================================
# Ejercicio 3
# =============================================================================
x = [2.36, 2.37, 2.38, 2.39]
fx = [0.85866, 0.86289, 0.86710, 0.87129]

h = 0.01 
# x = 2.36, que está en la posición 0
#(x + h) = 2.36 + 0.01 = 2.37, que está en la posición 1
#Usando diferencia forward
dfx = (fx[1] - fx[0]) / h
print("f'(0.2) ≈", round(dfx, 3))
print()

#(x + 2h) = 2.38, que está en la posición 3
d2fx = (fx[2] - 2*fx[1] + fx[0])/(h**2)
print("f''(0.2) ≈", round(d2fx, 3))
print()

#%%
# =============================================================================
# Ejercicio 4
# =============================================================================
'NO ME DIO TIEMPO :('
'''
Modulo regla trapezoidal recursiva

Inew = trapecio_recursiva(f,a,b,Iold,k).
Iold = Integral de f(x) de x = a hasta b calculada
con la regla trapezoidal recursiva con 2ˆ(k-1) paneles.
Inew = la misma integral calculada con 2ˆk paneles.

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
def f(x): return (math.sin(x))/math.sqrt(x) 
Iold = 0.0
for k in range(1,21):
  Inew = trapecio_recursiva(f,0.0,math.pi,Iold,k)
  if (k > 1) and (abs(Inew - Iold)) < 1.0e-6: break
  Iold = Inew

print('Integral =',Inew)
print('n Panels =',2**(k-1))
'''

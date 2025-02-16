import math
import cmath
import numpy as np
import sympy as sp
#%%
print('Ejercicio #1\n')
"""
-Define una función fahrenheit a celsius que toma un valor en Fahrenheit y lo 
convierte a Celsius usando la fórmula.
-En la función main, se solicita al usuario que ingrese una temperatura en 
Fahrenheit.
-Se llama a la función de conversión y se almacena el resultado.
-Se imprime la temperatura en Celsius, formateada a dos decimales.
"""
fahrenheit = float(input("Ingrese la temperatura en grados Fahrenheit: "))

celsius = (fahrenheit - 32) * 5 / 9

print(f'La temperatura en grados Celsius es: {round(celsius, 2)}°C')

#%%
print('Ejercicio #2')
"""
Considerando sinh(x)=(e^x - e^-x)/2

Elaborar un programa que calcule el valor de sinh(x) en x=2π de tres diferentes
maneras:
    1.Evaluando sinh(x) directamente.
    2.Evaluando con la definición del lado derecho, usando la función exponencial.
    3.Evaluando con la feninición del lado derecho, usando la potencia.
"""
#1) math.sinh
x=2*math.pi
sin1=math.sinh(x)
print(round(sin1,2))

#2) math.exp
sin2=(math.exp(x) - math.exp(-x))/2
print(round(sin2,2))

#3) math.e
sin3=(math.e**x - math.e**-x)/2
print(round(sin3,2))

#%%
print('Ejercicio 3')
"""
1.-Considera la relación entre el seno en variable compleja y el seno hiperbólico en
variable real x,
    
    sin(ix) = isinh(x)

    Elabora un programa que calcule el valor de sin(ix) y de sinh(x) para ciertos
    valores dados de x, para verificar la identidad.

2.-Considera la relación de Euler para x real,

    e^ix = cos(x) + isin(x)
    
    Elabora un programa que calcule el valor de cos(x), sin(x) y de e^ix para ciertos
    valores dados de x, para verificar la identidad.
"""
x = complex(input('Ingrese un número: '))

#Punto 1
print("Inciso 1")
sin_ix = cmath.sin(1j * x)
#Calcula el seno de un número imaginario 'ix' utilizando la función 'sin'.
i_sinh_x = 1j * cmath.sinh(x)
#Calcula 'i*sinh(x)', donde 'sinh' es la función seno hiperbólico, y '1j' representa la unidad imaginaria.
print(f"sin(ix) = {sin_ix}\nisinh(x) = {i_sinh_x}")
print(f"¿Son iguales? {cmath.isclose(sin_ix, i_sinh_x)}\n")
#Comprueba si los valores calculados de 'sin(ix)' e 'i * sinh(x)' son iguales.

#Punto 2
print("Inciso 2")
exp_ix = cmath.exp(1j * x)
#Calcula la exponencial de 'ix' usando 'exp'.
cos_sin = cmath.cos(x) + (1j * cmath.sin(x))
#Combina el coseno y el seno de 'x' para verificar la fórmula de Euler, utilizando '1j' para la parte imaginaria.
print(f"e^(ix) = {exp_ix}\ncos(x) + isin(x) = {cos_sin}")
print(f"¿Son iguales? {cmath.isclose(exp_ix, cos_sin)}\n")
#Verifica si la identidad de Euler se cumple, comparando 'e^(ix)' con 'cos(x) + i*sin(x)'.

#%%
print('Ejercicio 4\n')
"""
Este tratamiento flexible de funciones en el plano complejo permite encontrar
las raices reales o complejas, de una función cuadrática. Considera que las
raices de f(z) = az^2 + bz + c se obtienen:
    
    z^+- = (-b +- √(b^2 - 4ac)) / 2a

Elabora un programa en el que uses Numpy para calcular el valor de las raices
con diferentes valores dados de a, b, c, para obtener ejemplos de raices
reales y complejas.
"""
def calcular_raiz(a, b, c):
    discriminante = (b**2) - (4*a*c)
    raiz_discriminante = np.sqrt(discriminante + 0j)  # Permite números complejos
    
    z1 = (-b + raiz_discriminante) / (2 * a)
    z2 = (-b - raiz_discriminante) / (2 * a)
    
    return z1, z2

#Valores de (a, b, c)
a=float(input("Asigne un valor para a: "))
b=float(input("Asigne un valor para b: "))
c=float(input("Asigne un valor para c: "))

r1, r2 = calcular_raiz(a, b, c)
print(f"\nPara a={a}, b={b} y c={c}:\n    Raíces = {r1:.2f}, {r2:.2f}")

#%%
print('Ejercicio 5')
"""
¿Cuál es la trayectoria de una pelota que se lanza con una rapidéz inicial
v_0 a un ángulo θ medido de la horizontal?

Sabemos que la pelota seguirá una trayectoria y=f(x), donde, en ausencia de
resistencia del aire,

    f(x) = xtanθ - (g / 2*v_0^2)(x^2 / (cosθ)^2) + y_0
 

En esta expresión, x es la coordenada horizontal, g es la aceleración de la
gravedad y y_0 es la posición inicial de la pelota.

Elaborar un programa en el que se evalue la expresión. El programa debe 
escribir el valor de todas las variables involucradas junto con las unidades
usadas.
"""
v0, theta, g, x, y0 = sp.symbols('v0 theta g x y0') #Definir variables

f_x = x * sp.tan(theta) - (g / (2 * v0**2 * sp.cos(theta)**2)) * x**2 + y0 #Definir la funcion
print('\nLa ecuación de la trayectoria es: ')
sp.pprint(f_x)

valor_v0=20
valor_theta=30
valor_g=9.81
valor_y0=0
valor_x=5

valores = {v0:valor_v0, theta:sp.rad(valor_theta), g:valor_g, y0:valor_y0, x:valor_x} #Valores para variables
trayectoria_evaluada = f_x.subs(valores) #Sustituir valores
print(f'\nv_0={valor_v0} m/s\nθ={valor_theta}°\ng={valor_g} m/s^2\ny_0={valor_y0} m\nx={valor_x}')

print(f"\nLa trayectoria de la pelota evaluada en x={x} m es: {trayectoria_evaluada.evalf()} m")

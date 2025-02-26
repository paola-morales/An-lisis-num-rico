import numpy as np
from numpy.linalg import solve
#%%
""""
Ejercicio 1 - INTERSECCIÓN DE TRAYECTORIAS:
    Tres objetos se mueven de tal manera que sus trayectorias son:
        2x - y + 3z = 24
        2y - z = 14
        7x - 5y = 6
    Encontrar su punto de intersección.
"""
print('Ejercicio 1')
#Definimos las matrices
A = np.array([[2, -1, 3],
              [0, 2, -1],
               [7, -5, 0]])

B = np.array([24, 14, 6])

#Para resolver el sistema de ecuaciones
solución = solve(A, B)


x, y, z = solución
print(f"El punto de intersección es: x = {x}, y = {y}, z = {round(z,2)}\n")

#%%
"""
Ejercicio 2 - CARGA DE LOS QUARKS:
    Los protones y neutrones están formados cada uno por tres quarks. Los
    protones poseen dos quarks up (u) y un quark down (d), los neutrones
    poseen un quark up y dos quarks down. Si la carga de un protón es igual
    al positivo de la carga del electrón (+e) y la carga de un neutrón es 
    cero, determine las cargas de los quarks up y down. (Tip: suponga que
    +e=1).
"""
print('Ejercicio 2')
"Nuestro sistema de escuaciones quedaría como:"
    #x + 2y = 0
    #2x + 1 = 1
C = np.array([[1, 2],
              [2, 1]])

D = np.array([0, 1])

#Para resolver el sistema de ecuaciones
cargas = solve(C, D)

u, d = cargas
print(f'La carga del quark up es:{round(u,3)}, y la carga del quark down es: {round(d,3)}\n')

#%%
"""
Ejercicio 3 - METEOROS:
    El Centro de Investigación 1 examina la cantidad de meteoros que entran
    a la atmósfera. Con su equipo de recopilación de datos durante 8 horas
    captó 95kg de meteoros, por fuentes externas sabemos que fueron de 4
    distintas masas (1kg, 5kg, 10kg y 20kg). La cantidad total de meteoros
    fue de 26. Otro centro de investigación captó que la cantidad de meteoros
    de 5kg es 4 veces la cantidad de meteoros de 10kg, y el número de meteoros
    de 1kg es 1 menos que el doble de la cantidad de meteoros de 5kg. Después
    use matrices para encontrar el número asociado a cada masa de meteoros.
"""
print('Ejercicio 3')
#Definimos variables
'''
M1kg: número de meteoros de 1kg
M5kg: número de meteoros de 5kg
M10kg: número de meteoros de 10kg
M20kg: número de meteoros de 20kg
'''
#Creamos nuestras ecuaciones
'''
1*M1kg + 5*M5kg + 10*M10kg + 20*M20kg = 95
M1kg + M5kg + M10kg + M20kg = 26
M5kg = 4*M10kg   —>   M5kg - 4*M10kg = 0
M1kg = 2*M5kg - 1   —>   M1kg - 2*M5kg = -1
'''
#Creamos nuestras matrices
E=np.array([
    [1,1,1,1],    #M1kg + M5kg + M10kg + M20kg = 26
    [1,5,10,20],  #1*M1kg + 5*M5kg + 10*M10kg + 20*M20kg = 95
    [0,1,-4,0],   #M5kg - 4*M10kg = 0
    [1,-2,0,0]    #M1kg - 2*M5kg = -1
    ])
F=np.array([26,95,0,-1])

cant_meteoros=solve(E, F)

M1kg, M5kg, M10kg, M20kg = cant_meteoros
print(f'La cantidad de metoros detectados de 1kg es: {int(M1kg)}\nLa cantidad de metoros detectados de 5kg es: {int(M5kg)}\nLa cantidad de metoros detectados de 10kg es: {int(M10kg)}\nLa cantidad de metoros detectados de 20kg es: {int(M20kg)}\n')

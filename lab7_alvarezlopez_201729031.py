# Maria Sofia Alvarez Lopez - 201729031
# ESAI
# Laboratorio 7 - Programacion Cientifica

print('----------------------------------------')
print('Laboratorio 7 - Programacion Cientifica')
print('Maria Sofia Alvarez Lopez - 201729031')
print('----------------------------------------')

# %%
## Ejercicio 1 - Metodo de la biseccion
import numpy as np
''' Funcion que obtiene la raiz de una ecuacion no lineal de una variable usando el metodo de la biseccion.
params: 
f: funcion no lineal a la cual se busca encontrar su raiz.
intervalo: arreglo de numpy con el intervalo inicial en el cual se evaluara la funcion.
tol_x: tolerancia en x. 
tol_y: tolerancia en y. Por defecto, tol_x = tol_y = 10**-5.
n_max: numero maximo de iteraciones que le permito al algoritmo. Por defecto, n_max=1000.
retorna: 
num_iter: numero de iteraciones
x2: raiz final calculada
xr_iter: valor estimado de la raiz en cada iteracion
'''
def biseccion(f, intervalo, tol_x=10**-5, tol_y=10**-5, n_max=1000):
    num_iter = 0 # Inicializo una variable con el numero de iteraciones
    xr_iter = np.array([]) #Defino un arreglo para ir guardando los valores de las raices candidatas, obtenidas en cada iteracion
    x0, x1 = intervalo[0], intervalo[-1] #Obtengo los puntos inicial y final del intervalo que busco evaluar 
    x2_prev = x1 # Creo una variable inicial para la raiz en la iteracion anterior
    while 1: # Inicio un proceso iterativo
        num_iter += 1 # Se agrega una iteracion al numero total de iteraciones
        x2 = (x0+x1)/2 # Obtengo la raiz candidata
        xr_iter = np.append(xr_iter, x2) # Adiciono la raiz hallada al arreglo de raices obtenidas en cada iteracion
        # Se evaluan los criterios de parada:
        if np.abs(x2-x2_prev)<=tol_x: # Si la resta de las raiz anterior con la actual es menor que la tolerancia, termino el proceso
            return num_iter, x2, xr_iter
        if np.abs(f(x2))<=tol_y: # Si la funcion evaluada en la raiz actual es tan cercana a cero como la tolerancia lo pide, termino el proceso.
            return num_iter, x2, xr_iter
        # Reviso si las iteraciones que llevo estan por debajo del limite establecido.
        if num_iter >= n_max:
            print('Maximo numero de iteraciones')
            return num_iter, x2, xr_iter
        # Se actualiza el intervalo de acuerdo con la condicion
        if f(x0)*f(x2) <= 0: # La raiz se encuentra en el intervalo [x0, x2].
            x1 = x2
        else: # La raiz se encuentra en el intervalo [x2, x1].
            x0 = x2 
        # Se actualiza x2 antiguo con el actual para la siguiente iteracion
        x2_prev = x2

#%%
## Ejercicio 2 - Metodo de la falsa posicion 
import numpy as np
''' Funcion que obtiene la raiz de una ecuacion no lineal de una variable usando el metodo de la falsa posicion.
params: 
f: funcion no lineal a la cual se busca encontrar su raiz.
intervalo: arreglo de numpy con el intervalo inicial en el cual se evaluara la funcion.
tol_x: tolerancia en x. 
tol_y: tolerancia en y. Por defecto, tol_x = tol_y = 10**-5.
n_max: numero maximo de iteraciones que le permito al algoritmo. Por defecto, n_max=1000.
retorna: 
num_iter: numero de iteraciones
x2: raiz final calculada
xr_iter: valor estimado de la raiz en cada iteracion
'''
def falsa_posicion(f, intervalo, tol_x=10**-5, tol_y=10**-5, n_max=1000):
    num_iter = 0 # Inicializo una variable con el numero de iteraciones
    xr_iter = np.array([]) #Defino un arreglo para ir guardando los valores de las raíces candidatas, obtenidas en cada iteracion
    x0, x1 = intervalo[0], intervalo[-1] #Obtengo los puntos inicial y final del intervalo que busco evaluar 
    x2_prev = x1 # Creo una variable inicial para la raíz en la iteración anterior
    while 1: # Inicio un proceso iterativo
        num_iter += 1 # Se agrega una iteracion al numero total de iteraciones
        x2 = x1 - ((f(x1)*(x1-x0))/(f(x1)-f(x0))) # Se obtiene la raiz candidata
        xr_iter = np.append(xr_iter, x2) # Aniado la raiz al arreglo
        # Se evaluan los criterios de parada:
        if np.abs(x2-x2_prev)<=tol_x: # Si la resta de las raiz anterior con la actual es menor que la tolerancia, termino el proceso
            return num_iter, x2, xr_iter
        if np.abs(f(x2))<=tol_y: # Si la funcion evaluada en la raiz actual es tan cercana a cero como la tolerancia lo pide, termino el proceso.
            return num_iter, x2, xr_iter
        # Reviso si las iteraciones que llevo estan por debajo del limite establecido.
        if num_iter >= n_max:
            print('Maximo numero de iteraciones')
            return num_iter, x2, xr_iter
        # Se actualiza el intervalo de acuerdo con la condicion
        if f(x0)*f(x2) <= 0: # La raiz se encuentra en el intervalo [x0, x2].
            x1 = x2
        else: # La raiz se encuentra en el intervalo [x2, x1].
            x0 = x2 
        # Se actualiza x2 antiguo con el actual para la siguiente iteracion
        x2_prev = x2

#%%
## Ejercicio 3 - Metodo del punto fijo
import numpy as np
''' Funcion que obtiene la raiz de una ecuacion no lineal de una variable usando el metodo del punto fijo.
params: 
f: funcion no lineal a la cual se busca encontrar su raiz.
g: funcion necesaria para realizar el metodo del punto fijo.
x1: punto inicial.
tol_x: tolerancia en x. 
tol_y: tolerancia en y. Por defecto, tol_x = tol_y = 10**-5.
n_max: numero maximo de iteraciones que le permito al algoritmo. Por defecto, n_max=1000.
retorna: 
num_iter: numero de iteraciones
x2: raiz final calculada
xr_iter: valor estimado de la raiz en cada iteracion'''
def punto_fijo(f, g, x1, tol_x=10**-5, tol_y=10**-5, n_max=1000):
    num_iter = 0 # Inicializo una variable con el numero de iteraciones
    xr_iter = np.array([]) #Defino un arreglo para ir guardando los valores de las raices candidatas, obtenidas en cada iteracion
    x2_prev = x1 # Creo una variable inicial para la raiz en la iteracion anterior
    while 1: # Inicio un proceso iterativo
        num_iter += 1 # Se agrega una iteracion al numero total de iteraciones
        x2 = g(x1) # Se obtiene la raiz candidata
        xr_iter = np.append(xr_iter, x2) # Aniado la raiz al arreglo
        # Se evaluan los criterios de parada:
        if np.abs(x2-x2_prev)<=tol_x: # Si la resta de las raiz anterior con la actual es menor que la tolerancia, termino el proceso
            return num_iter, x2, xr_iter
        if np.abs(f(x2))<=tol_y: # Si la funcion evaluada en la raiz actual es tan cercana a cero como la tolerancia lo pide, termino el proceso.
            return num_iter, x2, xr_iter
        # Reviso si las iteraciones que llevo estan por debajo del limite establecido.
        if num_iter >= n_max:
            print('Maximo numero de iteraciones')
            return num_iter, x2, xr_iter
        # Se actualiza el intervalo de acuerdo con la condicion
        x1 = x2
        # Se actualiza x2 antiguo con el actual para la siguiente iteracion
        x2_prev = x2

#%%
## Ejercicio 4 - Metodo de Newton
# Primero, defino una funcion de ayuda para calcular la derivada
import numpy as np
from sympy import *
'''
Funcion que calcula la derivada de una funcion f.
params:
f: funcion a calcular su derivada
retorna:
num_iter: numero de iteraciones
x2: raiz final calculada
xr_iter: valor estimado de la raiz en cada iteracion'''
def derivative(f):
    x = Symbol('x')
    y_prima = f.diff(x)
    return lambdify(x, y_prima, 'numpy')

# Ahora si defino el metodo de Newton
''' Funcion que obtiene la raiz de una ecuacion no lineal de una variable usando el metodo de Newton.
params: 
f: funcion no lineal a la cual se busca encontrar su raiz.
f_sympy: Representacion de la funcion f usando sympy.
x1: punto inicial.
tol_x: tolerancia en x. 
tol_y: tolerancia en y. Por defecto, tol_x = tol_y = 10**-5.
n_max: numero maximo de iteraciones que le permito al algoritmo. Por defecto, n_max=1000.
retorna: 
num_iter: numero de iteraciones
x2: raiz final calculada
xr_iter: valor estimado de la raiz en cada iteracion
'''
def newton(f, f_sympy, x1,tol_x=10**-5, tol_y=10**-5, n_max=1000):
    num_iter = 0 # Inicializo una variable con el numero de iteraciones
    xr_iter = np.array([]) #Defino un arreglo para ir guardando los valores de las raices candidatas, obtenidas en cada iteracion
    x2_prev = x1 # Creo una variable inicial para la raiz en la iteracion anterior
    while 1: # Inicio un proceso iterativo
        num_iter += 1 # Se agrega una iteracion al numero total de iteraciones
        df = derivative(f_sympy)
        x2 = x1 - (f(x1)/df(x1)) # Se obtiene la raiz candidata
        xr_iter = np.append(xr_iter, x2) # Aniadimos la raiz al arreglo
        # Se evaluan los criterios de parada:
        if np.abs(x2-x2_prev)<=tol_x: # Si la resta de las raiz anterior con la actual es menor que la tolerancia, termino el proceso
            return num_iter, x2, xr_iter
        if np.abs(f(x2))<=tol_y: # Si la funcion evaluada en la raiz actual es tan cercana a cero como la tolerancia lo pide, termino el proceso.
            return num_iter, x2, xr_iter
        # Reviso si las iteraciones que llevo estan por debajo del limite establecido.
        if num_iter >= n_max:
            print('Maximo numero de iteraciones')
            return num_iter, x2, xr_iter
        # Se actualiza el intervalo de acuerdo con la condicion
        x1 = x2
        # Se actualiza x2 antiguo con el actual para la siguiente iteracion
        x2_prev = x2

#%%
## Ejercicio 5 - Metodo de la secante
import numpy as np
''' Funcion que obtiene la raiz de una ecuacion no lineal de una variable usando el metodo de la secante.
params: 
f: funcion no lineal a la cual se busca encontrar su raiz.
x0, x1: puntos iniciales sobre los cuales inicia el metodo.
tol_x: tolerancia en x. 
tol_y: tolerancia en y. Por defecto, tol_x = tol_y = 10**-5.
n_max: numero maximo de iteraciones que le permito al algoritmo. Por defecto, n_max=1000.
retorna: 
num_iter: numero de iteraciones
x2: raiz final calculada
xr_iter: valor estimado de la raiz en cada iteracion
'''
def secante(f, x0, x1,tol_x=10**-5, tol_y=10**-5, n_max=1000):
    num_iter = 0 # Inicializo una variable con el numero de iteraciones
    xr_iter = np.array([]) #Defino un arreglo para ir guardando los valores de las raices candidatas, obtenidas en cada iteracion
    x2_prev = x1 # Creo una variable inicial para la raiz en la iteración anterior
    while 1: # Inicio un proceso iterativo
        num_iter += 1 # Se agrega una iteracion al numero total de iteraciones
        x2 = x1 - ((f(x1)*(x1-x0))/(f(x1)-f(x0))) # Se obtiene la raiz candidata
        xr_iter = np.append(xr_iter, x2) # Aniado la raiz al arreglo
        # Se evaluan los criterios de parada:
        if np.abs(x2-x2_prev)<=tol_x: # Si la resta de las raiz anterior con la actual es menor que la tolerancia, termino el proceso
            return num_iter, x2, xr_iter
        if np.abs(f(x2))<=tol_y: # Si la funcion evaluada en la raiz actual es tan cercana a cero como la tolerancia lo pide, termino el proceso.
            return num_iter, x2, xr_iter
        # Reviso si las iteraciones que llevo estan por debajo del limite establecido.
        if num_iter >= n_max:
            print('Maximo numero de iteraciones')
            return num_iter, x2, xr_iter
        # Se actualiza el intervalo de acuerdo con la condicion
        x0, x1 = x1, x2
        # Se actualiza x2 antiguo con el actual para la siguiente iteracion
        x2_prev = x2

#%% Ejercicio 6 - Tasa de convergencia
'''Metodo que calcula la tasa de convergencia para un metodo de encontrar raices en particular a partir de un arreglo con los 
valores calculados en las iteraciones. Se usa el metodo visto en clase.
params:
x_r: El arreglo con las raices calculadas en las iteraciones.
f: Funcion a la cual se busca encontrar la raiz.
x1: Un punto inicial para encontrar la raiz, usando el metodo de scipy. 
retorna:
r_array: Un arreglo con las tasas de convergencia calculdas.'''
import scipy.optimize as opt
def tasa_convergencia(x_r, f, x1):
    # Estimamos el valor "verdadero" de la raíz a través de la función fsolve de scipy.
    xroot_true = opt.fsolve(f, x1)
    # Encuentro la estimacion del error para cada iteracion
    eps_array = np.abs(x_r - xroot_true)
    # Encuentro una estimacion de la tasa de convergencia para cada iteracion
    r_array = []
    r_array = ((np.log10(eps_array[1: np.size(eps_array)-1] / 
                     eps_array[2:np.size(eps_array)])) /
           (np.log10(eps_array[0: np.size(eps_array)-2] /
                     eps_array[1:np.size(eps_array)-1])))
    return r_array
    


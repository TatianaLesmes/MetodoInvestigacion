import numpy as np

# Definimos las funciones no lineales
def ecuacion1(V1, V2, V3):
    return V1**2 - V2 + V3 - 3

def ecuacion2(V1, V2, V3):
    return V1 - V2**3 + np.cos(V3) - 1

def ecuacion3(V1, V2, V3):
    return np.sin(V1) + V2 - V3**2

# Vector de funciones (las tres ecuaciones no lineales)
def sistema_ecuaciones(V):
    V1, V2, V3 = V
    return np.array([ecuacion1(V1, V2, V3), ecuacion2(V1, V2, V3), ecuacion3(V1, V2, V3)])

# Método de Broyden para resolver el sistema de ecuaciones no lineales
def metodo_broyden(sistema_ecuaciones, tensiones_iniciales, tolerancia=1e-6, max_iteraciones=200):
    num_variables = len(tensiones_iniciales)
    # Matriz Jacobiana inicial aproximada (identidad)
    jacobiano_aprox = np.eye(num_variables)
    tensiones = tensiones_iniciales
    for iteracion in range(max_iteraciones):
        # Evaluamos el sistema en el punto actual (tensiones)
        valores_funciones = sistema_ecuaciones(tensiones)
        
        # Verificamos si hemos llegado a una solución dentro de la tolerancia
        if np.linalg.norm(valores_funciones, ord=2) < tolerancia:
            return tensiones, iteracion  # Solución encontrada
        
        # Calculamos el cambio en las tensiones (V_k+1 = V_k - J_k^{-1} F(V_k))
        delta_tensiones = np.linalg.solve(jacobiano_aprox, -valores_funciones)
        nuevas_tensiones = tensiones + delta_tensiones
        
        # Evaluamos el sistema en las nuevas tensiones
        nuevos_valores_funciones = sistema_ecuaciones(nuevas_tensiones)
        cambio_funciones = nuevos_valores_funciones - valores_funciones
        
        # Actualizamos la matriz Jacobiana usando Broyden
        jacobiano_aprox += np.outer((cambio_funciones - jacobiano_aprox @ delta_tensiones), delta_tensiones) / np.dot(delta_tensiones, delta_tensiones)
        
        # Actualizamos las tensiones
        tensiones = nuevas_tensiones
    
    return tensiones, max_iteraciones  # Si no converge en el número máximo de iteraciones

# Valores iniciales para las tensiones
tensiones_iniciales = np.array([1.5, 1.5, 1.5])

# Ejecutamos el método de Broyden
solucion_final, num_iteraciones = metodo_broyden(sistema_ecuaciones, tensiones_iniciales)

# Mostramos el resultado detallado
resultado = f"Solucion encontrada:\nTension V1 = {solucion_final[0]:.4f}\nTension V2 = {solucion_final[1]:.4f}\nTension V3 = {solucion_final[2]:.4f}\n"
iteraciones_resultado = f"Numero de iteraciones: {num_iteraciones}\n"

# Usamos print para mostrar en consola
print(resultado)
print(iteraciones_resultado)
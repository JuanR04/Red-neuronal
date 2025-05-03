import numpy as np

# Datos de entrada (grados centígrados)
X = np.array([-10, 0, 10, 20, 30, 40, 50, 60, 70, 80])
X_min, X_max = np.min(X), np.max(X)
X = (X - X_min) / (X_max - X_min)  # Escalar entre 0 y 1

# Datos de salida (grados Fahrenheit)
Y = np.array([14, 32, 50, 68, 86, 104, 122, 140, 158, 176])
Y_min, Y_max = np.min(Y), np.max(Y)
Y = (Y - Y_min) / (Y_max - Y_min)  # Escalar entre 0 y 1

epocas = 20000
tasa_aprendizaje = 0.01  # Aumentada para mejorar la convergencia

# Inicialización de los pesos y sesgos con un rango mayor
p11, p12 = np.random.uniform(-0.5, 0.5, 2)
p21, p22 = np.random.uniform(-0.5, 0.5, 2)
s11, s12, s21 = 0.1, 0.1, 0.1

# Entrenamiento del modelo
for i in range(epocas):
    error_total = 0
    for j in range(len(X)):
        suma1 = X[j] * p11 + s11
        suma2 = X[j] * p12 + s12
        sigma1 = 1 / (1 + np.exp(-suma1))  # Sigmoide para neurona 1
        sigma2 = 1 / (1 + np.exp(-suma2))  # Sigmoide para neurona 2

        salida = sigma1 * p21 + sigma2 * p22 + s21
        error = Y[j] - salida
        error_total += error**2

        # Actualización de pesos y sesgos
        p21 += tasa_aprendizaje * error * sigma1
        p22 += tasa_aprendizaje * error * sigma2
        s21 += tasa_aprendizaje * error
        p11 += tasa_aprendizaje * error * p21 * sigma1 * (1 - sigma1) * X[j]
        p12 += tasa_aprendizaje * error * p22 * sigma2 * (1 - sigma2) * X[j]
        s11 += tasa_aprendizaje * error * p21 * sigma1 * (1 - sigma1)
        s12 += tasa_aprendizaje * error * p22 * sigma2 * (1 - sigma2)

    if i % 100 == 0:
        print(f"Época {i}, Error total: {error_total:.6f}")

# Función para hacer predicciones
def predecir(valor):
    valor_normalizado = (valor - X_min) / (X_max - X_min)
    suma1 = valor_normalizado * p11 + s11
    suma2 = valor_normalizado * p12 + s12
    sigma1 = 1 / (1 + np.exp(-suma1))
    sigma2 = 1 / (1 + np.exp(-suma2))
    salida_normalizada = sigma1 * p21 + sigma2 * p22 + s21
    return salida_normalizada * (Y_max - Y_min) + Y_min

# Ciclo de predicción
while True:
    valor_entrada = float(input("Ingresa un valor en grados centígrados para predecir (o 'q' para salir): "))
    prediccion = predecir(valor_entrada)
    print(f"La predicción para {valor_entrada} °C es: {prediccion:.2f} °F")
    
    continuar = input("¿Deseas hacer otra predicción? (s/n): ").strip().lower()
    if continuar != 's':
        print("Fin del programa.")
        break

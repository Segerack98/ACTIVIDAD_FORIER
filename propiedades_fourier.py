
# ==================================================
# PROPIEDADES DE LA TRANSFORMADA DE FOURIER
# ==================================================

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. CONFIGURACION GENERAL
# --------------------------------------------------

fs = 1000
duracion = 2
t = np.arange(0, duracion, 1 / fs)
N = len(t)

# Crear carpeta donde guardar las imagenes
carpeta = Path(__file__).resolve().parent / "graficas"
carpeta.mkdir(exist_ok=True)

# Funcion para calcular la transformada y magnitud
def calcular_fft(senal):
    X = np.fft.fft(senal)
    frecuencias = np.fft.fftfreq(N, d=1 / fs)

    X = np.fft.fftshift(X)
    frecuencias = np.fft.fftshift(frecuencias)

    magnitud = np.abs(X) / N
    return frecuencias, magnitud

# --------------------------------------------------
# 2. PROPIEDAD DE LINEALIDAD
# --------------------------------------------------

# Dos senales con diferentes frecuencias
x1 = np.sin(2 * np.pi * 5 * t)
x2 = 0.5 * np.sin(2 * np.pi * 12 * t)

# Sumar las senales
suma = x1 + x2

# Calcular transformadas
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)
X_suma = np.fft.fft(suma)

# Verificar linealidad numericamente
error_linealidad = np.max(np.abs(
    X_suma - (X1 + X2)
))

print("PROPIEDAD DE LINEALIDAD")
print("Error maximo:", error_linealidad)

# Calcular espectros para graficar
f, mag1 = calcular_fft(x1)
_, mag2 = calcular_fft(x2)
_, mag_suma = calcular_fft(suma)

fig, ax = plt.subplots(2, 1, figsize=(11, 7))

ax[0].plot(t, x1, label="Senal de 5 Hz")
ax[0].plot(t, x2, label="Senal de 12 Hz")
ax[0].plot(t, suma, label="Suma", alpha=0.7)
ax[0].set_xlim(0, 1)
ax[0].set_title("Linealidad - Dominio del tiempo")
ax[0].set_xlabel("Tiempo (s)")
ax[0].set_ylabel("Amplitud")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(f, mag1, label="5 Hz")
ax[1].plot(f, mag2, label="12 Hz")
ax[1].plot(f, mag_suma, "--", label="Suma")
ax[1].set_xlim(-20, 20)
ax[1].set_title("Linealidad - Espectros")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Magnitud normalizada")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig(carpeta / "propiedad_linealidad.png",
            dpi=300)

# --------------------------------------------------
# 3. PROPIEDAD DE DESPLAZAMIENTO TEMPORAL
# --------------------------------------------------

# Senal original de 5 Hz
original = np.sin(2 * np.pi * 5 * t)

# Desplazar 0.05 segundos
retraso = 0.05
muestras = int(retraso * fs)

# Desplazamiento circular exacto para la DFT
desplazada = np.roll(original, muestras)

# FFT de ambas senales
X_original = np.fft.fft(original)
X_desplazada = np.fft.fft(desplazada)

# Factor de fase teorico
f_fft = np.fft.fftfreq(N, d=1 / fs)
factor = np.exp(-1j * 2 * np.pi * f_fft * retraso)

# Verificar propiedad sobre numeros complejos
error_desplazamiento = np.max(np.abs(
    X_desplazada - X_original * factor
))

print("\nPROPIEDAD DE DESPLAZAMIENTO")
print("Error maximo:", error_desplazamiento)

# Obtener magnitudes
f, mag_original = calcular_fft(original)
_, mag_desplazada = calcular_fft(desplazada)

# Calcular cambio de fase en 5 Hz
indice_5 = np.argmin(np.abs(f_fft - 5))

cambio_fase = np.angle(
    X_desplazada[indice_5] /
    X_original[indice_5]
)

print("Cambio de fase en 5 Hz:",
      cambio_fase, "radianes")

fig, ax = plt.subplots(3, 1, figsize=(11, 9))

ax[0].plot(t, original, label="Original")
ax[0].plot(t, desplazada, "--", label="Retrasada")
ax[0].set_xlim(0, 0.6)
ax[0].set_title("Desplazamiento - Tiempo")
ax[0].set_xlabel("Tiempo (s)")
ax[0].set_ylabel("Amplitud")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(f, mag_original, label="Original")
ax[1].plot(f, mag_desplazada, "--",
           label="Retrasada")
ax[1].set_xlim(-15, 15)
ax[1].set_title("Magnitudes de Fourier")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Magnitud normalizada")
ax[1].legend()
ax[1].grid(True)

# Fase en +5 Hz para comparar ambas senales
fase_original = np.angle(X_original[indice_5])
fase_desplazada = np.angle(
    X_desplazada[indice_5]
)

ax[2].bar(
    ["Original", "Retrasada"],
    [fase_original, fase_desplazada]
)
ax[2].set_title("Comparacion de fase en 5 Hz")
ax[2].set_ylabel("Fase (radianes)")
ax[2].set_ylim(-np.pi - 0.3, np.pi + 0.3)
ax[2].grid(True, axis="y")

plt.tight_layout()
plt.savefig(carpeta / "propiedad_desplazamiento.png",
            dpi=300)

# --------------------------------------------------
# 4. PROPIEDAD DE ESCALAMIENTO
# --------------------------------------------------

# Senal original
senal_5 = np.sin(2 * np.pi * 5 * t)

# Comprimir el tiempo por un factor de 2
a = 2
senal_10 = np.sin(2 * np.pi * 5 * a * t)

# Transformadas
f, mag_5 = calcular_fft(senal_5)
_, mag_10 = calcular_fft(senal_10)

# Frecuencia positiva con maxima magnitud
positivas = f > 0

pico_5 = f[positivas][np.argmax(mag_5[positivas])]
pico_10 = f[positivas][np.argmax(mag_10[positivas])]

print("\nPROPIEDAD DE ESCALAMIENTO")
print("Frecuencia original:", pico_5, "Hz")
print("Frecuencia escalada:", pico_10, "Hz")
print("Relacion de frecuencias:", pico_10 / pico_5)

fig, ax = plt.subplots(2, 1, figsize=(11, 7))

ax[0].plot(t, senal_5, label="Original 5 Hz")
ax[0].plot(t, senal_10, "--", label="Escalada 10 Hz")
ax[0].set_xlim(0, 0.6)
ax[0].set_title("Escalamiento - Tiempo")
ax[0].set_xlabel("Tiempo (s)")
ax[0].set_ylabel("Amplitud")
ax[0].legend()
ax[0].grid(True)

ax[1].plot(f, mag_5, label="Original 5 Hz")
ax[1].plot(f, mag_10, "--", label="Escalada 10 Hz")
ax[1].set_xlim(-20, 20)
ax[1].set_title("Escalamiento - Frecuencia")
ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_ylabel("Magnitud normalizada")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.savefig(carpeta / "propiedad_escalamiento.png",
            dpi=300)

# --------------------------------------------------
# 5. MOSTRAR RESULTADOS
# --------------------------------------------------

print("\nSimulacion finalizada correctamente.")
print("Imagenes guardadas en:", carpeta)

plt.show()

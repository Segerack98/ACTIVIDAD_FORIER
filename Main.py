

# ACTIVIDAD FORMATIVA 2
# SIMULACION Y ANALISIS CON TRANSFORMADA DE FOURIER



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# 1. CONFIGURACION DE LA SIMULACION


fs = 1000               # Frecuencia de muestreo (Hz)
duracion = 2            # Duracion de la senal (s)
t = np.arange(0, duracion, 1 / fs)

N = len(t)              # Numero total de muestras


# 2. GENERACION DE LAS TRES SEÑALES


# Señal senoidal de 5 Hz
frecuencia = 5
senoidal = np.sin(2 * np.pi * frecuencia * t)

# Pulso rectangular de 0.5 a 1 segundo
pulso = np.where((t >= 0.5) & (t < 1.0), 1, 0)

# Escalon que se activa en el segundo 1
escalon = np.where(t >= 1.0, 1, 0)


# 3. FUNCION PARA CALCULAR LA FFT


def calcular_fourier(senal):

    # Calcular la Transformada Rapida de Fourier
    transformada = np.fft.fft(senal)

    # Obtener las frecuencias correspondientes
    frecuencias = np.fft.fftfreq(N, d=1 / fs)

    # Centrar el espectro: frecuencias negativas
    # a la izquierda y positivas a la derecha
    transformada = np.fft.fftshift(transformada)
    frecuencias = np.fft.fftshift(frecuencias)

    # Normalizar la magnitud por numero de muestras
    magnitud = np.abs(transformada) / N

    # Calcular la fase en radianes
    fase = np.angle(transformada)

    # Ocultar la fase cuando la magnitud es pequeña
    umbral = 0.001 * np.max(magnitud)
    fase[magnitud < umbral] = np.nan

    return frecuencias, magnitud, fase


# 4. CALCULO DE FOURIER PARA CADA SEÑAL


freq_sen, mag_sen, fase_sen = calcular_fourier(senoidal)

freq_pul, mag_pul, fase_pul = calcular_fourier(pulso)

freq_esc, mag_esc, fase_esc = calcular_fourier(escalon)


# 5. FUNCION PARA CREAR LAS GRAFICAS


def graficar_senal(t, senal, freq, mag, fase, nombre,
                  archivo):

    plt.figure(figsize=(12, 9))

    # Grafica 1: Dominio del tiempo
    plt.subplot(3, 1, 1)
    plt.plot(t, senal, color="blue")
    plt.title(nombre + " - Dominio del tiempo")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)

    # Grafica 2: Magnitud del espectro
    plt.subplot(3, 1, 2)
    plt.plot(freq, mag, color="red")
    plt.title("Magnitud de la Transformada de Fourier")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud normalizada")
    plt.xlim(-30, 30)
    plt.grid(True)

    # Grafica 3: Fase del espectro
    plt.subplot(3, 1, 3)
    plt.plot(freq, fase, ".", color="green")
    plt.title("Fase de la Transformada de Fourier")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Fase (radianes)")
    plt.xlim(-30, 30)
    plt.ylim(-np.pi - 0.2, np.pi + 0.2)
    plt.grid(True)

    plt.tight_layout()

    # Guardar figura
    plt.savefig(archivo, dpi=300)

    # Mostrar figura
    plt.show()


# 6. CREAR CARPETA PARA LOS RESULTADOS


carpeta = Path("graficas")
carpeta.mkdir(exist_ok=True)


# 7. GRAFICAR LAS TRES SEÑALES


graficar_senal(
    t, senoidal, freq_sen, mag_sen, fase_sen,
    "Senal senoidal de 5 Hz",
    carpeta / "fourier_senoidal.png"
)

graficar_senal(
    t, pulso, freq_pul, mag_pul, fase_pul,
    "Pulso rectangular",
    carpeta / "fourier_pulso.png"
)

graficar_senal(
    t, escalon, freq_esc, mag_esc, fase_esc,
    "Funcion escalon",
    carpeta / "fourier_escalon.png"
)

print("Transformadas de Fourier calculadas.")
print("Graficas guardadas en la carpeta graficas.")

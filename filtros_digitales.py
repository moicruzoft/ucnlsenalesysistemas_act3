#-------------------------------------------------------------------------------
# Name:        Filtros
# Purpose:
#
# Author:      moises cruz cruz
#
# Created:     30/07/2025
# Copyright:   (c) moise 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# Configuración de gráficos
#plt.style.use('seaborn')
plt.style.use('seaborn-v0_8')  # En versiones recientes
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

def generar_senal_compuesta(frecuencias, amplitudes, duracion=1.0, fs=1000):
    """
    Genera una señal compuesta por múltiples componentes de frecuencia

    Parámetros:
    - frecuencias: Lista de frecuencias en Hz
    - amplitudes: Lista de amplitudes para cada frecuencia
    - duracion: Duración en segundos
    - fs: Frecuencia de muestreo en Hz

    Retorna:
    - t: Vector de tiempo
    - x: Señal compuesta
    """
    t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)
    x = np.zeros_like(t)

    for freq, amp in zip(frecuencias, amplitudes):
        x += amp * np.sin(2 * np.pi * freq * t)

    # Añadir ruido blanco
    ruido = np.random.normal(0, 0.2, x.shape)
    x += ruido

    return t, x

# Generar señal de prueba
frecuencias = [10, 50, 100]  # Hz
amplitudes = [1.0, 0.5, 0.3]
fs = 1000  # Frecuencia de muestreo
t, senal = generar_senal_compuesta(frecuencias, amplitudes)

# Graficar señal original
plt.figure()
plt.plot(t, senal)
plt.title('Señal original con ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.show()


def analizar_espectro(senal, fs):
    """
    Calcula y grafica el espectro de frecuencia de una señal

    Parámetros:
    - senal: Señal a analizar
    - fs: Frecuencia de muestreo

    Retorna:
    - freqs: Vector de frecuencias
    - magnitud: Magnitud del espectro
    """
    n = len(senal)
    yf = fft(senal)
    freqs = fftfreq(n, 1/fs)[:n//2]
    magnitud = np.abs(yf[0:n//2]) / n

    plt.figure()
    plt.plot(freqs, 20 * np.log10(magnitud))
    plt.title('Espectro de frecuencia de la señal')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud [dB]')
    plt.grid(True)
    plt.xlim(0, fs/2)
    plt.show()

    return freqs, magnitud

# Analizar espectro de la señal original
freqs, magnitud = analizar_espectro(senal, fs)


def disenar_filtro_butterworth(frec_corte, fs, orden=5, tipo='low'):
    """
    Diseña un filtro Butterworth IIR

    Parámetros:
    - frec_corte: Frecuencia de corte (o lista para bandpass/bandstop)
    - fs: Frecuencia de muestreo
    - orden: Orden del filtro
    - tipo: 'low', 'high', 'bandpass', 'bandstop'

    Retorna:
    - b, a: Coeficientes del filtro
    """
    nyq = 0.5 * fs
    normal_cutoff = np.array(frec_corte) / nyq

    b, a = signal.butter(orden, normal_cutoff, btype=tipo, analog=False)
    return b, a

# Diseñar filtro pasa bajos Butterworth
frec_corte = 30  # Hz
b_butter, a_butter = disenar_filtro_butterworth(frec_corte, fs, orden=4, tipo='low')

# Graficar respuesta en frecuencia del filtro
w, h = signal.freqz(b_butter, a_butter, worN=8000)
plt.figure()
plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
plt.title('Respuesta en frecuencia del filtro Butterworth')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Ganancia [dB]')
plt.grid(True)
plt.axvline(frec_corte, color='red', linestyle='--', label='Frecuencia de corte')
plt.legend()
plt.xlim(0, 100)
plt.ylim(-60, 5)
plt.show()



def disenar_filtro_fir(frec_corte, fs, numtaps=101, tipo='low'):
    """
    Diseña un filtro FIR usando el método de ventana

    Parámetros:
    - frec_corte: Frecuencia de corte (o lista para bandpass/bandstop)
    - fs: Frecuencia de muestreo
    - numtaps: Número de coeficientes (orden + 1)
    - tipo: 'low', 'high', 'bandpass', 'bandstop'

    Retorna:
    - coeficientes: Coeficientes del filtro FIR
    """
    nyq = 0.5 * fs
    cutoff = np.array(frec_corte) / nyq

    coeficientes = signal.firwin(numtaps, cutoff, window='hamming', pass_zero=tipo)
    return coeficientes

# Diseñar filtro FIR pasa bajos
coef_fir = disenar_filtro_fir(frec_corte, fs, numtaps=101, tipo='lowpass')

# Graficar respuesta en frecuencia del filtro FIR
w, h = signal.freqz(coef_fir, worN=8000)
plt.figure()
plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)))
plt.title('Respuesta en frecuencia del filtro FIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Ganancia [dB]')
plt.grid(True)
plt.axvline(frec_corte, color='red', linestyle='--', label='Frecuencia de corte')
plt.legend()
plt.xlim(0, 100)
plt.ylim(-60, 5)
plt.show()




def aplicar_filtro(senal, b, a=None):
    """
    Aplica un filtro IIR o FIR a una señal

    Parámetros:
    - senal: Señal a filtrar
    - b: Coeficientes del numerador
    - a: Coeficientes del denominador (None para FIR)

    Retorna:
    - senal_filtrada: Señal después del filtrado
    """
    if a is None:  # Es un filtro FIR
        senal_filtrada = signal.lfilter(b, [1.0], senal)
    else:  # Es un filtro IIR
        senal_filtrada = signal.filtfilt(b, a, senal)

    return senal_filtrada

# Aplicar filtro Butterworth
senal_butter = aplicar_filtro(senal, b_butter, a_butter)

# Aplicar filtro FIR
senal_fir = aplicar_filtro(senal, coef_fir)

# Graficar comparación
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, senal)
plt.title('Señal original')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, senal_butter)
plt.title('Señal filtrada con Butterworth (IIR)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, senal_fir)
plt.title('Señal filtrada con FIR')
plt.grid(True)

plt.tight_layout()
plt.show()

# Analizar espectros después del filtrado
analizar_espectro(senal_butter, fs)
analizar_espectro(senal_fir, fs)




def comparar_filtros():
    # Crear señal de prueba con más componentes
    frecuencias = [5, 30, 80, 120, 200]
    amplitudes = [1.0, 0.7, 0.5, 0.3, 0.2]
    t, senal = generar_senal_compuesta(frecuencias, amplitudes, duracion=0.5, fs=1000)

    # Diseñar filtros
    frec_corte = 50  # Hz

    # Butterworth IIR
    b_butter, a_butter = disenar_filtro_butterworth(frec_corte, fs, orden=6, tipo='low')

    # FIR con ventana
    coef_fir = disenar_filtro_fir(frec_corte, fs, numtaps=151, tipo='lowpass')

    # Aplicar filtros
    senal_butter = aplicar_filtro(senal, b_butter, a_butter)
    senal_fir = aplicar_filtro(senal, coef_fir)

    # Graficar resultados
    plt.figure(figsize=(12, 10))

    # Señales en tiempo
    plt.subplot(3, 1, 1)
    plt.plot(t, senal)
    plt.title('Señal original')
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, senal_butter)
    plt.title('Filtro Butterworth IIR (orden 6)')
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, senal_fir)
    plt.title('Filtro FIR (150 coeficientes)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Respuesta en frecuencia
    plt.figure(figsize=(12, 6))

    # Butterworth
    w, h = signal.freqz(b_butter, a_butter, worN=8000)
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)), label='Butterworth IIR')

    # FIR
    w, h = signal.freqz(coef_fir, worN=8000)
    plt.plot(0.5*fs*w/np.pi, 20*np.log10(np.abs(h)), label='FIR')

    plt.title('Comparación de respuestas en frecuencia')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Ganancia [dB]')
    plt.grid(True)
    plt.axvline(frec_corte, color='red', linestyle='--', label='Frecuencia de corte')
    plt.legend()
    plt.xlim(0, 150)
    plt.ylim(-80, 5)
    plt.show()

comparar_filtros()
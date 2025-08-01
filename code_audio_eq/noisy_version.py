import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# === Parameters ===
samp_freq = 1000.0             # Sample rate in Hz
notch_freq = 50.0              # Frequency to be removed (Hz)
quality_factor = 20.0          # Quality factor (Q)

# === Design notch filter ===
b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)

# === Frequency response of the notch filter ===
w, h = signal.freqz(b_notch, a_notch, fs=samp_freq)
notch_index = np.argmin(np.abs(h))
actual_notch_freq = w[notch_index]

# === Plot magnitude response ===
plt.figure(figsize=(8, 6))
plt.plot(w, 20 * np.log10(np.abs(h)), 'b', label='Notch filter response')
plt.axvline(actual_notch_freq, color='red', linestyle='--',
            label=f'Notch at {50} Hz')
plt.xlabel('Frequency [Hz]', fontsize=14)
plt.ylabel('Magnitude [dB]', fontsize=14)
plt.title('Frequency Response of Notch Filter', fontsize=16)
plt.grid()
plt.legend()
plt.xlim(0, 100)  # Zoom in to see the notch clearly
plt.tight_layout()

# === Generate synthetic signal ===
t = np.linspace(0, 1.0, int(samp_freq), endpoint=False)  # 1 sec duration
signal_f1 = np.sin(2 * np.pi * 15 * t)                  # 15 Hz
signal_f2 = np.sin(2 * np.pi * 50 * t)                  # 50 Hz (to be removed)
noise = 0.03 * np.random.normal(0, 1, t.shape[0])       # Small Gaussian noise

noisy_signal = signal_f1 + signal_f2 + noise

# === Apply the notch filter ===
filtered_signal = signal.filtfilt(b_notch, a_notch, noisy_signal)

# === Plot signals before and after filtering ===
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, noisy_signal, label='Noisy Signal', color='red')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Signal Before Filtering')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(t, filtered_signal, label='Filtered Signal', color='blue')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Signal After 50 Hz Notch Filtering')
plt.grid()

plt.tight_layout()
plt.show() # Shows all the generated figures

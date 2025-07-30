import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# --- Parameters ---
duration = 3.0  # In seconds
samplerate = 44100 # Orginally 44100 Hz
f0 = 1000.0  # Frequency to remove (500 or 1000 or 3000)
Q = 30.0    # Quality factor (originally 30)

# --- Generate synthetic signal ---
t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
signal_orig = (
    0.5 * np.sin(2 * np.pi * 500 * t) +
    0.5 * np.sin(2 * np.pi * 1000 * t) +  # Target for the notch
    0.3 * np.sin(2 * np.pi * 3000 * t)
)
signal_orig /= np.max(np.abs(signal_orig))  # Normalize

# --- Design and apply notch filter ---
b, a = signal.iirnotch(f0, Q, fs=samplerate)
signal_filtered = signal.lfilter(b, a, signal_orig)

# --- Save the created audios ---
sf.write("synthetic_original.wav", signal_orig, samplerate)
sf.write("synthetic_filtered.wav", signal_filtered, samplerate)

# Select hardware output speaker from the computer
sd.default.device = (None, 3)  # (input_device, output_device)

# --- Playback ---
print("Playing original...")
sd.play(signal_orig, samplerate)
sd.wait()

time.sleep(2) # Waits 2 seconds

print("Playing filtered...")
sd.play(signal_filtered, samplerate)
sd.wait()

# --- Frequency domain plot ---
print("Plotting figure...")
N = len(signal_orig)
xf = fftfreq(N, 1 / samplerate)
yf_orig = np.abs(fft(signal_orig))
yf_filt = np.abs(fft(signal_filtered))

plt.figure(figsize=(12, 3)) # Adjust graph size
plt.plot(xf[:N//2], yf_orig[:N//2], label='Original')
plt.plot(xf[:N//2], yf_filt[:N//2], label='Filtered')

for freq in [500, 1000, 3000]:
    plt.axvline(freq, color='gray', linestyle='--', alpha=0.5)
    plt.text(freq + 20, max(yf_orig[:N//2]) * 0.5, f'{freq} Hz', color='gray')

plt.xlim(250, 3250)  # Zoom in on main frequencies
plt.title(f'Frequency Spectrum : Original vs Filtered\nFrequency to remove : {f0} Hz ; Quality factor : {Q} ; Sample rate : {samplerate} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.tight_layout()

# --- Notch filter plot ---
w, h = signal.freqz(b, a, fs=samplerate)
plt.figure()
plt.plot(w, 20 * np.log10(abs(h)))
plt.title('Notch Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.xlim(0, 5000)
plt.axvline(f0, color='red', linestyle='--', label=f'Notch at {f0} Hz')
plt.legend()
plt.tight_layout()

# Shows all of the generated figures at once
plt.show()

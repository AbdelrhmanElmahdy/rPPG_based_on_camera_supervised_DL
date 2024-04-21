import numpy as np
import scipy
import scipy.io
from scipy.signal import butter
from scipy.sparse import spdiags


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _calculate_fft_hr(ppg_signal, fs=60, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def moving_heart_rate(x_peaks, window, fs):
    hr=[]
    x=[]
    x_peaks=np.array(x_peaks)
    l=len(x_peaks)-(window-1)
    for r in range(l):
        x.clear()
        for i in range(window):
            x.append(x_peaks[r+i])
        hr.append(((window-1)*60)/((x[-1]-x[0]+1)/fs))
    return hr
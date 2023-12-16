# DAFX book 7.7
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program performs time-frequency filtering
# after calculation of the FIR (here bandpass)

import numpy as np

import scipy.signal
from scipy.io import wavfile

# User data
s_FIR       = 1280
s_win       = 2*s_FIR

fs, wav_in  = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in     = np.array(wav_in) * 1.0
DAFx_in     = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in * np.iinfo(np.int16).max).astype(np.int16))

# Init
L           = len(DAFx_in)
DAFx_in_pad = np.append(DAFx_in, np.zeros(s_win-np.mod(L, s_FIR)))
DAFx_out    = np.zeros(s_win + int(np.ceil(len(DAFx_in_pad))))

grain       = np.zeros(s_win)
vec_pad     = np.zeros(s_FIR)

fr          = 1000/fs
alpha       = -0.002
fir         = [np.exp(alpha*x) * np.sin(2*np.pi*fr*x) for x in range(s_FIR)]
fir2        = np.append(fir, np.zeros(s_win-s_FIR))
fcorr       = np.fft.fft(fir2)

pin         = 0
pend        = len(DAFx_in_pad) - s_FIR
while pin < pend:
    grain       = np.append(DAFx_in_pad[pin:pin+s_FIR], vec_pad)
    # compute input wave FFT
    f           = np.fft.fft(grain) * fcorr
    grain       = np.real(np.fft.ifft(f))
    DAFx_out[pin:pin+s_win] = DAFx_out[pin:pin+s_win] + grain

    pin         = pin + s_FIR


# Write out as .wav
DAFx_out    = DAFx_out[s_win:]
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out))

wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

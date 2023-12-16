# DAFX book 7.15
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program performs integer ratio time stretching
#  using the FFT-IFFT approach

import numpy as np

import scipy.signal
from scipy.io import wavfile


# User data
n1          = 512
n2          = n1
WLen        = 2048
w1          = scipy.signal.get_window('hann', WLen, fftbins=True)
w2          = w1

fs, wav_in  = wavfile.read('pvoc/sounds/violin_scale.wav')
DAFx_in1    = np.array(wav_in) * 1.0

fs, wav_in  = wavfile.read('pvoc/sounds/whalesong.wav')
DAFx_in2    = np.array(wav_in) * 1.0


# Init
L            = min(len(DAFx_in1), len(DAFx_in2))
DAFx_in1_pad = np.append(np.zeros(WLen), DAFx_in1)
DAFx_in1_pad = np.append(DAFx_in1_pad, np.zeros(WLen-np.mod(L, n1)))
DAFx_in1_pad = DAFx_in1_pad / max(abs(DAFx_in1))

DAFx_in2_pad = np.append(np.zeros(WLen), DAFx_in2)
DAFx_in2_pad = np.append(DAFx_in2_pad, np.zeros(WLen-np.mod(L, n1)))
DAFx_in2_pad = DAFx_in2_pad / max(abs(DAFx_in2))


DAFx_out     = np.zeros(WLen + int(np.ceil(L)))

pin         = 0
pout        = 0
pend        = L - WLen
while pin < pend:
    grain1      = DAFx_in1_pad[pin:pin+WLen]*w1
    grain2      = DAFx_in2_pad[pin:pin+WLen]*w1
    # compute input wave FFT
    f1          = np.fft.fft(np.fft.fftshift(grain1))
    r1          = np.abs(f1)
    theta1      = np.angle(f1)
    f2          = np.fft.fft(np.fft.fftshift(grain2))
    r2          = np.abs(f2)
    theta2      = np.angle(f2)

    #  the next two lines can be changed according to the effect
    r           = r1
    theta       = theta2
    ft          = r * np.exp(1j*theta)
    grain       = np.fft.fftshift(np.real(np.fft.ifft(ft)))*w2

    DAFx_out[pout:pout+WLen] = DAFx_out[pout:pout+WLen] + grain

    pin         = pin + n1
    pout        = pout + n2


# Write out as .wav
DAFx_out = DAFx_out[WLen:] * np.iinfo(np.int16).max / max(abs(DAFx_out))
wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

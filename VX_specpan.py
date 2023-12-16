# DAFX book 7.19
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program makes a spectral panning of a sound

import numpy as np

import scipy.signal
from scipy.io import wavfile


# User data
n1          = 512
n2          = n1
s_win       = 2048
fs, wav_in  = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in     = np.array(wav_in) * 1.0
DAFx_in     = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in* np.iinfo(np.int16).max).astype(np.int16))

# Init
w1          = scipy.signal.get_window('hann', s_win, fftbins=True)
w2          = w1
L           = len(DAFx_in)
DAFx_in_pad = np.append(np.zeros(s_win), DAFx_in)
DAFx_in_pad = np.append(DAFx_in_pad, np.zeros(s_win-np.mod(L, n1)))
DAFx_out    = np.zeros((s_win + int(np.ceil(len(DAFx_in_pad))), 2))
hs_win      = int(s_win/2)
coef        = np.sqrt(2)/2

# control: clipped sine wave with a few periods; in [-pi/4;pi/4]
theta       = [2*np.sin(200*n/s_win) for n in range(hs_win)]
theta       = [min(1, max(-1, t)) * np.pi/4 for t in theta]

# control: rough left/right split at Fs/30 (for Fs=44.1kS/s, ~=1470 Hz)
#theta       = [np.pi/4 if ((x/2) < hs_win/30) else -np.pi/4 for x in range(hs_win)]

# preserving phase symmetry
theta       = np.append(theta[:], theta[::-1])

pin         = 0
pout        = 0
pend        = len(DAFx_in_pad) - s_win
while pin < pend:
    grain       = DAFx_in_pad[pin:pin+s_win]*w1

    f           = np.fft.fft(grain)
    # compute left and right spectrum with Blumlein law at 45◦
    ftL         = coef * f * (np.cos(theta) + np.sin(theta))
    ftR         = coef * f * (np.cos(theta) - np.sin(theta))
    grainL      = (np.real(np.fft.ifft(ftL))) * w2
    grainR      = (np.real(np.fft.ifft(ftR))) * w2

    DAFx_out[pout:pout+s_win, 0] = DAFx_out[pout:pout+s_win, 0] + grainL
    DAFx_out[pout:pout+s_win, 1] = DAFx_out[pout:pout+s_win, 1] + grainR

    pin         = pin + n1
    pout        = pout + n2


# Write out as .wav
DAFx_out    = DAFx_out[s_win:]
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out.flatten()))

wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

# DAFX book 7.17
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program makes the whisperization of a sound, by randomizing the phase

import numpy as np

import scipy.signal
from scipy.io import wavfile

# User data
s_win       = 512
n1          = int(s_win/8)
n2          = n1
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
DAFx_out    = np.zeros(s_win + int(np.ceil(len(DAFx_in_pad))))


pin         = 0
pout        = 0
pend        = len(DAFx_in_pad) - s_win
while pin < pend:
    grain       = DAFx_in_pad[pin:pin+s_win]*w1
    # compute input wave FFT
    f           = np.fft.fft(np.fft.fftshift(grain))
    r           = np.abs(f)
    phi         = 2*np.pi*np.random.rand(s_win,1)
    ft          = r * np.exp(1j*phi)
    grain       = np.fft.fftshift(np.real(np.fft.ifft(r)))*w2
    DAFx_out[pout:pout+s_win] = DAFx_out[pout:pout+s_win] + grain

    pin         = pin + n1
    pout        = pout + n2



# Write out as .wav
DAFx_out    = DAFx_out[s_win:]
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out))

wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

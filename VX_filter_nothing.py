# DAFX book 7.2
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program (i) performs a complex-valued filter bank
# then (ii) filters a sound through the filter bank
# and (iii) reconstructs a sound


import numpy as np

import scipy.signal
from scipy.io import wavfile

# User data
s_win       = 256
nChannel    = 128
n1          = 1024
fs, wav_in  = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in     = np.array(wav_in) * 1.0
DAFx_in     = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in* np.iinfo(np.int16).max).astype(np.int16))

# Init windows, arrays, etc.
window      = scipy.signal.get_window('hann', s_win, fftbins=True)
L           = len(DAFx_in)
DAFx_in_pad = np.append(DAFx_in, np.zeros(n1))

DAFx_out    = np.empty((1,1))
X_tilde     = np.zeros((n1, nChannel), dtype = 'complex_')
z           = np.zeros((s_win-1, nChannel), dtype = 'complex_')

# Init complex BPF bank
filt        = np.zeros((s_win, nChannel), dtype = 'complex_')
t_filt      = range(-int(s_win/2), int(s_win/2))
for k in range(nChannel):
    wk          = 2*np.pi*1j*(k/s_win)
    filt[:, k]  = window * [np.exp(wk*t) for t in t_filt]

pin         = 0
pend        = len(DAFx_in_pad) - n1
while pin < pend:
    grain       = DAFx_in_pad[pin:pin+n1]
    for k in range(nChannel):
        X_tilde[:, k], z[:, k] = scipy.signal.lfilter(filt[:, k], 1, grain, axis=-1, zi=z[:, k])

    pin     = pin + n1

    # Reconstruct from BPF outputs
    res1    = np.real(np.sum(X_tilde, axis=1))
    DAFx_out= np.append(DAFx_out, res1)



# Write out as .wav
DAFx_out    = DAFx_out[nChannel+1:nChannel+1+L] / nChannel
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out))

wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

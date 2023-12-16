# DAFX book 7.1
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program (i) implements a heterodyne filter bank,
# then (ii) filters a sound through the filter bank
# and (iii) reconstructs a sound

import numpy as np

import scipy.signal
from scipy.io import wavfile

# User data
s_win       = 256
nChannel    = 128
s_block     = 1024
fs, wav_in  = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in     = np.array(wav_in) * 1.0
DAFx_in     = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in* np.iinfo(np.int16).max).astype(np.int16))

# Init windows, arrays, etc.
window      = scipy.signal.get_window('hann', s_win, fftbins=True)
s_buffer    = len(DAFx_in)
DAFx_in_pad = np.append(DAFx_in, np.zeros(s_block))

DAFx_out    = np.empty((1,1))
X           = np.zeros((s_block, nChannel), dtype = 'complex_')
X_tilde     = np.zeros((s_block, nChannel), dtype = 'complex_')
z           = np.zeros((s_win-1, nChannel), dtype = 'complex_')

# Init heterodyne filters
t_het       = range(s_block)
het         = np.zeros((s_block, nChannel), dtype = 'complex_')
het2        = np.zeros((s_block, nChannel), dtype = 'complex_')
for k in range(nChannel):
    wk          = 2*np.pi*1j*((k+1)/s_win)
    het[:, k]   = [np.exp(wk*(t+s_win/2)) for t in t_het]
    het2[:, k]  = [np.exp(-1*wk*t) for t in t_het]

pin         = 0
pend        = len(DAFx_in_pad) - s_block
while pin < pend:
    grain       = DAFx_in_pad[pin:pin+s_block]
    for k in range(nChannel):
        X[:, k], z[:, k] = scipy.signal.lfilter(window, 1, grain*het[:, k], axis=-1, zi=z[:, k])
        X_tilde          = X*het2

    pin     = pin + s_block

    # Reconstruct from BPF outputs
    res1    = np.real(np.sum(X_tilde, axis=1))
    DAFx_out= np.append(DAFx_out, res1)


# Write out as .wav
DAFx_out    = DAFx_out[nChannel+1:nChannel+1+s_buffer] / nChannel
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out))

wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

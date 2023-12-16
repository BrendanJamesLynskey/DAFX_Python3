# DAFX book 7.5
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program performs signal convolution with gaborets


# import sys
# from os.path import dirname, join as pjoin

import numpy as np

import scipy.signal
from scipy.io import wavfile

def princarg(phs):
    """ Return 'principal argument' of unwrapped phase, i.e. in range ]-pi,pi] """
    return np.mod(phs + np.pi, -2*np.pi) + np.pi

# User data
n1          = 128
n2          = n1
s_win       = 512
fs, wav_in  = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in     = np.array(wav_in) * 1.0
DAFx_in     = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in* np.iinfo(np.int16).max).astype(np.int16))


# Init
window      = scipy.signal.get_window('hann', s_win, fftbins=True)

nChannel    = int(s_win/2)
L           = len(DAFx_in)
DAFx_in_pad = np.append(np.zeros(s_win), DAFx_in)
DAFx_in_pad = np.append(DAFx_in_pad, np.zeros(s_win-np.mod(L, n1)))/ max(np.abs(DAFx_in))
DAFx_out    = np.zeros(len(DAFx_in_pad))


# 512-sample Gaboret for each channel
t_gab       = range(-int(s_win/2), int(s_win/2))
gab         = np.zeros((nChannel, s_win), dtype = 'complex_')   # (256, 512) array of Gaborets
for k in range(nChannel):
    wk          = 2*np.pi*1j*((k+1)/s_win)
    gab[k, :]   = window * [np.exp(wk*t) for t in t_gab]

pin         = 0
pout        = 0
pend        = len(DAFx_in_pad) - s_win

vec         = np.zeros(256, dtype = 'complex_')     # x256 array (vector) of inner-products (1 for each channel)
res         = np.zeros(s_win)                       # (512, 1) array of reconstructed audio for this time-slice


while pin < pend:
    grain       = DAFx_in_pad[pin:pin+s_win]

    # Complex vector corresponding to vertical line
    vec         = np.dot(gab, grain)

    # Reconstruction from vector to grain
    res         = np.real(np.transpose(np.conj(gab)) @ vec)

    DAFx_out[pout+1:pout+s_win+1] = DAFx_out[pout+1:pout+s_win+1] + res
    # DAFx_out[pout:pout+s_win] = DAFx_out[pout:pout+s_win] + grain@np.identity(len(grain))

    pin     = pin + n1
    pout    = pout + n2


# Write out as .wav
DAFx_out = DAFx_out[s_win : s_win + L] * np.iinfo(np.int16).max / max(DAFx_out)

wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

# DAFX book 7.13
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program performs pitch shifting
#  using the FFT-IFFT approach

import numpy as np

import scipy.signal
from scipy.io import wavfile

def princarg(phs):
    """ Return 'principal argument' of unwrapped phase, i.e. in range ]-pi,pi] """
    return np.mod(phs + np.pi, -2*np.pi) + np.pi

# User data
n2          = 512
pit_ratio   = 1.2
s_win       = 2048
fs, wav_in  = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in     = np.array(wav_in) * 1.0
DAFx_in     = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in* np.iinfo(np.int16).max).astype(np.int16))

n1          = int(np.round(n2 / pit_ratio))
tstretch_ratio  = n2/n1

# Init
w1          = scipy.signal.get_window('hann', s_win, fftbins=True)
w2          = w1
L           = len(DAFx_in)
DAFx_in_pad = np.append(np.zeros(s_win), DAFx_in)
DAFx_in_pad = np.append(DAFx_in_pad, np.zeros(s_win-np.mod(L, n1)))
DAFx_out    = np.zeros(s_win + int(np.ceil(len(DAFx_in_pad))))

omega       = np.array([2*np.pi*n1*i/s_win for i in range(s_win)]) # bin Fcentre vector
phi0        = np.zeros(s_win)  # bin last grain's FFT phase
psi         = np.zeros(s_win)  # bin interpolated (accumulated) phase

# For linear interpolation of a grain of length s_win
lx          = int(np.floor(s_win*n1/n2))
x           = np.arange(lx) * s_win/lx
ix          = np.floor(x)
dx          = x - ix
dx1         = 1 - dx

grain3      = np.zeros(s_win)

pin         = 0
pout        = 0
pend        = len(DAFx_in_pad) - s_win
while pin < pend:
    grain       = DAFx_in_pad[pin:pin+s_win]*w1
    # compute input wave FFT
    f           = np.fft.fft(np.fft.fftshift(grain))
    r           = np.abs(f)
    phi         = np.angle(f)

    delta_phi   = omega + princarg(phi - phi0 - omega)
    phi0        = np.copy(phi)
    psi         = princarg(psi+delta_phi*tstretch_ratio)

    # compute output-grain spectrum, with time-stretched instantaneous freq
    ft          = r*np.exp(1j*phi)
    # synthesise output grain gen, store in output wave
    grain       = np.fft.fftshift(np.real(np.fft.ifft(ft)))*w2
    grain2      = np.append(grain, 0)
    for n, ig in enumerate(ix):
        grain3[n]  = grain2[int(ig)] * dx1[n] + grain2[int(ig+1)] * dx[n]

    DAFx_out[pout:pout+s_win] = DAFx_out[pout:pout+s_win] + grain3

    pin         = pin + n1
    pout        = pout + n1



# Write out as .wav
DAFx_out    = DAFx_out[s_win:]
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out))
wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))


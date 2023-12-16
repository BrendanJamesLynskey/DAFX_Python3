# DAFX book 7.12
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program performs pitch shifting
# using the oscillator bank approach

import numpy as np

import scipy.signal
from scipy.io import wavfile

def princarg(phs):
    """ Return 'principal argument' of unwrapped phase, i.e. in range ]-pi,pi] """
    return np.mod(phs + np.pi, -2*np.pi) + np.pi

# User data
n1          = 512
pit_ratio   = 1.2
s_win       = 2048
fs, wav_in  = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in     = np.array(wav_in) * 1.0
DAFx_in     = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in* np.iinfo(np.int16).max).astype(np.int16))

w1          = scipy.signal.get_window('hann', s_win, fftbins=True)
w2          = w1
L           = len(DAFx_in)

DAFx_in_pad = np.append(np.zeros(s_win), DAFx_in)
DAFx_in_pad = np.append(DAFx_in_pad, np.zeros(s_win-np.mod(L, n1)))
DAFx_out    = np.zeros(s_win + int(np.ceil(len(DAFx_in_pad))))


hs_win      = int(s_win/2)
omega       = np.array([2*np.pi*n1*i/s_win for i in range(hs_win)]) # bin Fcentre vector
phi0        = np.zeros(hs_win)  # bin last grain's FFT phase
r0          = np.zeros(hs_win)  # bin interpolated (accumulated) mag
psi         = np.zeros(hs_win)  # bin interpolated (accumulated) phase
grain       = np.zeros(s_win)
res         = np.zeros(n1)


pin         = 0
pout        = 0
pend        = len(DAFx_in_pad) - s_win
while pin < pend:
    grain       = DAFx_in_pad[pin:pin+s_win]*w1

    fc          = np.fft.fft(np.fft.fftshift(grain), norm='forward')
    f           = fc[:hs_win] * 2
    r           = np.abs(f)
    phi         = np.angle(f)

    # Unwrapped phase diff on each bin for this analysis step
    #   Expressed as heterodyned phase, +/- some offset from Fcentre
    delta_phi   = omega + princarg(phi - phi0 - omega)

    # Phase and mag inc per sample, to be used for linear interp and reconstruction
    delta_r     = (r-r0)/n1
    delta_psi   = pit_ratio*delta_phi/n1

    # Compute sum of weighted cosine
    #   for each bin, accumulate delta r and psi, hitting values from FFT at end
    for k in range(n1):
        r0          = r0 + delta_r              # accumulate mag over output period
        psi         = psi + delta_psi           # accum phase over output period
        res[k]      = np.sum(r0 * np.cos(psi))  # sum all contributions from spectral bands

    # Although now phi0=phi and r0=0, reset to results from FFT ready for next pass
    #   ensures same phase quadrant, and original FP precision
    phi0        = np.copy(phi)
    r0          = np.copy(r)
    # Prepare phi for next interpolation(accumulation), from ~0 phase shift
    psi         = princarg(psi)

    DAFx_out[pout:pout+n1] = res

    pin     = pin + n1
    pout    = pout + n1


# Write out as .wav
DAFx_out = DAFx_out[int(s_win/2)+n1:]
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out))
wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))


# DAFX book 7.10
# Adapted from https://dafx.de/DAFX_Book_Page_2nd_edition/matlab.html
#   Basic structure and variable names were kept

# This program performs real ratio time stretching using the
# FFT-IFFT approach, applying spectral peak phase-locking

import numpy as np

import scipy.signal
from scipy.io import wavfile

def princarg(phs):
    """ Return 'principal argument' of unwrapped phase, i.e. in range ]-pi,pi] """
    return np.mod(phs + np.pi, -2*np.pi) + np.pi

# User data
n1              = 256
n2              = 300
s_win           = 2048  # Good frequency resolution needed to identify peaks
fs, wav_in      = wavfile.read('pvoc/sounds/test_000.wav')
DAFx_in         = np.array(wav_in) * 1.0
DAFx_in         = DAFx_in / max(abs(DAFx_in))
wavfile.write('pvoc/DAFX_ip.wav', fs, (DAFx_in* np.iinfo(np.int16).max).astype(np.int16))

tstretch_ratio  = n2/n1
hs_win          = int(s_win/2)

# Init windows, arrays, etc.
w1              = scipy.signal.get_window('hann', s_win, fftbins=True)
w2              = scipy.signal.get_window('hann', s_win, fftbins=True)
L               = len(DAFx_in)
DAFx_in_pad     = np.append(np.zeros(s_win), DAFx_in)
DAFx_in_pad     = np.append(DAFx_in_pad, np.zeros(s_win-np.mod(L, n1)))
DAFx_out        = np.zeros(s_win + int(np.ceil(len(DAFx_in_pad)*tstretch_ratio)))

omega           = np.array([2*np.pi*n1*i/s_win for i in range(s_win)])
phi0            = np.zeros(hs_win+1)
psi             = np.zeros(hs_win+1)
psi2            = np.zeros(hs_win+1)
nprevpeaks      = 0
prev_peak_loc   = np.zeros(hs_win+1)

pin             = 0
pout            = 0
pend            = len(DAFx_in_pad) - s_win
while pin < pend:

    grain           = DAFx_in_pad[pin:pin+s_win]*w1

    # Compute input wave FFT; input real, so compute only first half
    f               = np.fft.rfft(grain)
    r               = np.abs(f)
    phi             = np.angle(f)

    # Find spectral peaks
    peak_loc        = np.zeros(hs_win+1)
    npeaks          = 0

    for b in range(2, hs_win-1):
        if r[b]>r[b-1] and r[b]>r[b-2] and r[b]>r[b+1] and r[b]>r[b+2]:
            peak_loc[npeaks]    = b
            npeaks              += 1
            b                   = b + 3

    # Propagate peak phases and compute spectral bin phases
    if pin == 0:
        psi     = np.copy(phi)
    elif npeaks>0 and nprevpeaks>0:
        prev_p  = 0
        for p in range(npeaks):
            p2      = int(peak_loc[p])
            # Connect current peak to the previous closest peak
            while ( prev_p < nprevpeaks and abs(p2-prev_peak_loc[prev_p+1]) < abs(p2-prev_peak_loc[prev_p]) ):
                prev_p  += 1
            p1                  = int(prev_peak_loc[prev_p])

            # Propagate peak's phase assuming linear frequency variation between connected peaks p1 and p2
            avg_p               = (p1 + p2) * 0.5
            pomega              = 2 * np.pi * n1 * avg_p / s_win
            peak_delta_phi      = pomega + princarg(phi[p2]-phi0[p1]-pomega)
            peak_target_phase   = princarg(psi[p1] + peak_delta_phi*tstretch_ratio)
            peak_phase_rotation = princarg(peak_target_phase-phi[p2])

            # Rotate phases of all bins around the current peak
            if npeaks==1:
                bin1                = 0
                bin2                = hs_win
            elif p==0:
                bin1                = 0
                bin2                = hs_win
            elif p==npeaks-1:
                bin1                = int(np.round((peak_loc[p-1]+p2)*0.5))
                bin2                = hs_win
            else:
                bin1                = int(np.round((peak_loc[p-1]+p2)*0.5))+1
                bin2                = int(np.round((peak_loc[p+1]+p2)*0.5))

            psi2[bin1:bin2+1]   = princarg(phi[bin1:bin2+1] + peak_phase_rotation)

        psi         = np.copy(psi2)

    else:
        delta_phi   = omega[:hs_win+1] + princarg(phi-phi0-omega[:hs_win+1])
        psi         = princarg(psi+delta_phi*tstretch_ratio)

    ft              = r*np.exp(1j*psi)

    # Reconstruct spectrum
    ft              = np.fft.irfft(ft)
    grain           = ft * w2

    DAFx_out[pout:pout+s_win] = DAFx_out[pout:pout+s_win] + grain

    # Store values for next frame
    phi0            = np.copy(phi)
    prev_peak_loc   = np.copy(peak_loc)
    nprevpeaks      = npeaks
    pin             = pin + n1
    pout            = pout + n2


# Write out as .wav
DAFx_out    = DAFx_out[s_win:]
DAFx_out    = DAFx_out * np.iinfo(np.int16).max / max(abs(DAFx_out))

wavfile.write('pvoc/DAFX_op.wav', fs, DAFx_out.astype(np.int16))

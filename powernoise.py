# encoding: utf-8
# Powernoise.py
# Adapted by Henrique Castilho 2019-04-14
# From the original:
########################################################
#       Arbitrary Spectral Slope Noise Generation      #
#               with MATLAB Implementation             #
#                                                      #
# From Little, 1992. Version by R. Rosa   08/08/14     #
########################################################

import numpy as np
import math

def powernoise(beta, N, *varargin):
# Generate samples of power law noise. The power spectrum
# of the signal scales as f^(-beta).
#
# Usage:
#  x = powernoise(beta, N)
#  x = powernoise(beta, N, 'option1', 'option2', ...)
#
# Inputs:
#  beta  - power law scaling exponent
# For instance:
# white noise   -> beta = 0;
# pink noise    -> beta = -1;
# red noise     -> beta = -2;
#
#  N     - number of samples to generate
#
# Output:
#  x     - N x 1 vector of power law samples
#
# With no option strings specified, the power spectrum is
# deterministic, and the phases are uniformly distributed in the range
# -pi to +pi. The power law extends all the way down to 0Hz (DC)
# component. By specifying the 'randpower' option string however, the
# power spectrum will be stochastic with Chi-square distribution. The
# 'normalize' option string forces scaling of the output to the range
# [-1, 1], consequently the power law will not necessarily extend
# right down to 0Hz.
#
# (cc) Max Little, 2008. This software is licensed under the
# Attribution-Share Alike 2.5 Generic Creative Commons license:
# http://creativecommons.org/licenses/by-sa/2.5/
# If you use this work, please cite:
# Little MA et al. (2007), "Exploiting nonlinear recurrence and fractal
# scaling properties for voice disorder detection", Biomed Eng Online, 6:23
#
# As of 20080323 markup
# If you use this work, consider saying hi on comp.dsp
# Dale B. Dalrymple 

    opt_randpow = False
    opt_normal = False

    for arg in varargin:
        if arg == 'normalize':
            opt_normal = True
        if arg == 'randpower':
            opt_randpow = True

    N2 = int(N / 2) - 1
    f = np.arange(2, (N2 + 1) + 1, 1)
    A2 = 1.0 / (f ** (beta / 2.0))

    if not opt_randpow:
        p2 = (np.random.uniform(0, 1, N2) - 0.5) * 2 * math.pi
        d2 = A2 * np.exp(1j * p2)
    else:
        # 20080323
        p2 = np.random.rand(N2) + 1j * np.random.rand(N2)
        d2 = A2 * p2

    d = np.concatenate(([1], d2, [1.0/((N2 + 2.0) ** beta)], np.flipud(np.conjugate(d2))))
    x = np.real(np.fft.ifft(d))

    if opt_normal:
        x = ((x - min(x)) / (max(x) - min(x)) - 0.5) * 2

    return x

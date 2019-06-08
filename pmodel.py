# Conversion from Mathlab to Python:
# http://www2.meteo.uni-bonn.de/staff/venema/themes/surrogates/pmodel/pmodel.m

# *** function varargout = pmodel(noValues, p, slope) ***
# This function generates a multifractal time series using the p-model.
# Optionally you can also filter the result from the p-model in Fourier
# space to give it another fractal slope, e.g. to make it continuous and
# nonstationary. This is also called fractional integration. The p-model 
# itself can only produce stationary time series, i.e time series where 
# the variance is finite if you would extrapolate its power spectrum to 
# infinite large scales.
# 
# The parameter of the p-model is p. With p values close to 1 or 0 the time
# series is very peaked. With values close to 0.5 the p-model is much
# calmer; p=0.5 results in a constant unity vector.
# The parameter for the fractal integration is the slope of the power
# spectrum. Davis et al. calls slopes flatter than -1 stationary, and
# slopes between -1 and -3 nonstationary, with stationary increments. 
# These nonstationary cases are at least continuous, but not
# differentiable. Slopes steeper than -3 are nonstionary and
# differentiable.
# 
# Possible calls:
# [pModelTimeSeries] = pmodel
# [pModelTimeSeries] = pmodel(noValues)
# [pModelTimeSeries] = pmodel(noValues, p)
# [fractionalIntegratedTimeSeries] = pmodel(noValues, p, slope)
# [fractionalIntegratedTimeSeries, pModelTimeSeries] = pmodel(noValues, p, slope)
# If no number of values (noValues) is specified or it is empty, the
# default value of 256 is used.
# If no p is specified or it is empty, the default value of 0.375 is used.
# If no slope is specified, you will get a time series without fractional
# integration.
# If a slope is specified you will get the fractionally integrated time
# series, or if you use two output variables you will get both the p-model
# time series as well as its fractionally integrated version.
#
# This Matlab function is based on the article: Davis, A., A. Marshak, R.
# Cahalan, and W. Wiscombe, The landsat scale break in stratocumulus as a
# three-dimensional radiative transfer effect: implications for cloud
# remote sensing. Journal of the Atmospheric Sciences, Vol. 54, no. 2, 1997.
# First version by Victor Venema, Victor.Venema@uni-bonn.de, 25 January 2006.


# NumPy is the fundamental package for scientific computing with Python
import numpy as np
# This module provides access to the math functions defined by the C standard
import math


# function y2 = next_step_1d(y, p)
def next_step_1d(y, p):                     
    length = len(y)                         # len = numel(y);
    y2 = np.zeros(length * 2)               # y2 = zeros(1,len*2);
    sign = np.random.uniform(0, 1, length) - 0.5  # sign=rand(1, len)-0.5;
    sign = sign / abs(sign)                 # sign = sign./abs(sign);
    y2[::2] = y + sign * (1 - 2 * p) * y    # y2(1:2:2*len-1)=y+sign.*(1-2*p).*y;
    y2[1::2] = y - sign * (1 - 2 * p) * y   # y2(2:2:2*len)  =y-sign.*(1-2*p).*y;
    return y2


#function a=fractal_spectrum_1d(noValues, slope)
def fractal_spectrum_1d(noValues, slope):
# If you want to make a large number of time series, please rewrite this
# part to get rid of the for-loop. :-)
    ori_vector_size = noValues              # ori_vector_size = noValues;
    ori_half_size = int(ori_vector_size / 2)  # ori_half_size=ori_vector_size/2;
    # The magnitudes of the Fourier coefficients
    a = np.zeros(ori_vector_size)           # a = zeros(ori_vector_size,1);
    for t2 in range(1, (ori_half_size+1)+1, 1):   # for t2 = 1:ori_half_size+1
        index = t2 - 1                      # index = t2-1;
        t4 = 2 + ori_vector_size - t2       # t4 = 2 + ori_vector_size - t2;
        if t4 > ori_vector_size :           # if ( t4 > ori_vector_size )
            t4 = t2                         # t4 = t2;
        with np.errstate(divide='ignore') :
            coeff = np.array([index])**slope  # coeff = index.^slope;
        a[t2-1] = coeff                     # a(t2) = coeff;
        a[t4-1] = coeff                     # a(t4) = coeff;
    a[0] = 0    # The DC-component of the Fourier spectrum should be zero
    return a


def pmodel(**kwargs) :
    noValues = kwargs.get('noValues')
    p = kwargs.get('p')
    slope = kwargs.get('slope')

    # Check input
    # if ( nargin < 1 | isempty(noValues) )
    if len(kwargs) < 1 or kwargs.get('noValues') == 'None' :
        noValues = 256                  # noValues = 256;
    # if ( nargin < 2 | isempty(p) )
    if len(kwargs) < 2 or kwargs.get('p') == 'None' :
        p = 0.375                       # p = 0.375;
    if len(kwargs) < 3 :                # if ( nargin < 3 )
        slope = []                      # slope = [];

    # Calculate length of time series
    # noOrders=ceil(log2(noValues));
    noOrders = int(math.ceil(math.log(noValues, 2)))

    # noValuesGenerated = 2.^(noOrders); <== not used
    
    # y is the time series generated with the p-model.
    y = [1]                             # y = 1;
    for n in range(1, noOrders + 1, 1): # for n=1:noOrders
        y = next_step_1d(y, p)          # y = next_step_1d(y, p);

    # If a slope if specified also a fractionally integrated time series(x)
    # is calculated from y.
    if not slope == [] :                # if ( ~isempty(slope) )
        # Calculate the magnitudes of the coefficients of the Fourier spectrum.
        # The Fourier slope is half of the slope of the power spectrum.
        # fourierCoeff = fractal_spectrum_1d(noValues, slope/2)';
        fourierCoeff = fractal_spectrum_1d(noValues, slope / 2.0)
        meanVal = np.mean(y)            # meanVal = mean(y);
        stdy = np.std(y)                # stdy = std(y);
        # Calculate the Fourier coefficients of the original p-model time series
        x = np.fft.ifft(y - meanVal)    # x = ifft(y-meanVal);
        # Calculate the phases, as these are kept intact, should not be changed
        # by the Fourier filter
        phase = np.angle(x)             # phase = angle(x);
        # Calculate the complex Fourier coefficients with the specified
        # spectral slope, and the phases of the p-model
        x = fourierCoeff * np.exp(1j * phase)  #x = fourierCoeff .* exp(i*phase);
        # Generate the fractionally integrated time series.
        x = np.real(np.fft.fft(x))      # x = real(fft(x));
        x = x * stdy / np.std(x)        # x = x * stdy / std(x);
        x = x + meanVal                 # x = x + meanVal;
    else:
        x = y                           # x = y;

# not implented:
# Reduce the sizes of the time series and put them in the right output variable.
# if (nargout == 1 )
#    y=y(1:noValues);
#    varargout{1} = x;
# else
#    y=y(1:noValues);
#    x=x(1:noValues);
#    varargout{1} = x;
#    varargout{2} = y;
# end

    x = x[0:noValues+1]
    return x


# REFERENCES
# https://stackoverflow.com/questions/25087769/runtimewarning-divide-by-zero-error-how-to-avoid-python-numpy/25088239
# https://stackoverflow.com/questions/20161899/scipy-curve-fit-error-divide-by-zero-encountered
# https://docs.scipy.org
# https://pastebin.com/CxxVrwZF accessed 2019-05-31
# http://www2.meteo.uni-bonn.de/staff/venema/themes/surrogates/pmodel/pmodel.m
# https://www.python.org/
# https://www.gnu.org/software/octave/

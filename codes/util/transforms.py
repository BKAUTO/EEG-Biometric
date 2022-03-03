import torch
import numpy as np
import copy
import scipy.signal as signal
#-----------------------------#
from pyts.image import MarkovTransitionField
from scipy.signal import firwin, butter, lfilter


class filterBank(object):

    """
    filter the given signal in the specific bands using cheby2 iir filtering.
    If only one filter is specified then it acts as a simple filter and returns 2d matrix
    Else, the output will be 3d with the filtered signals appended in the third dimension.
    axis is the time dimension along which the filtering will be applied
    """

    def __init__(self, filtBank, fs, filtAllowance=2, axis=1, filtType='filter'):
        self.filtBank = filtBank
        self.fs = fs
        self.filtAllowance =filtAllowance
        self.axis = axis
        self.filtType=filtType

    def bandpassFilter(self, data, bandFiltCutF,  fs, filtAllowance=2, axis=1, filtType='filter'):
        """
         Filter a signal using cheby2 iir filtering.

        Parameters
        ----------
        data: 2d/ 3d np array
            trial x channels x time
        bandFiltCutF: two element list containing the low and high cut off frequency in hertz.
            if any value is specified as None then only one sided filtering will be performed
        fs: sampling frequency
        filtAllowance: transition bandwidth in hertz
        filtType: string, available options are 'filtfilt' and 'filter'

        Returns
        -------
        dataOut: 2d/ 3d np array after filtering
            Data after applying bandpass filter.
        """
        aStop = 30 # stopband attenuation
        aPass = 3 # passband attenuation
        nFreq= fs/2 # Nyquist frequency
        
        if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
            # no filter
            print("Not doing any filtering. Invalid cut-off specifications")
            return data
        
        elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
            # low-pass filter
            print("Using lowpass filter since low cut hz is 0 or None")
            fPass =  bandFiltCutF[1]/ nFreq
            fStop =  (bandFiltCutF[1]+filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'lowpass')
        
        elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
            # high-pass filter
            print("Using highpass filter since high cut hz is None or nyquist freq")
            fPass =  bandFiltCutF[0]/ nFreq
            fStop =  (bandFiltCutF[0]-filtAllowance)/ nFreq
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'highpass')
        
        else:
            # band-pass filter
            # print("Using bandpass filter")
            fPass =  (np.array(bandFiltCutF)/ nFreq).tolist()
            fStop =  [(bandFiltCutF[0]-filtAllowance)/ nFreq, (bandFiltCutF[1]+filtAllowance)/ nFreq]
            # find the order
            [N, ws] = signal.cheb2ord(fPass, fStop, aPass, aStop)
            b, a = signal.cheby2(N, aStop, fStop, 'bandpass')

        if filtType == 'filtfilt':
            dataOut = signal.filtfilt(b, a, data, axis=axis)
        else:
            dataOut = signal.lfilter(b, a, data, axis=axis)
        return dataOut

    def __call__(self, data1):
        data = copy.deepcopy(data1)

        # initialize output
        out = np.zeros([*data.shape, len(self.filtBank)])

        # repetitively filter the data.
        for i, filtBand in enumerate(self.filtBank):
            out[:,:,i] = self.bandpassFilter(data, filtBand, self.fs, self.filtAllowance,
                    self.axis, self.filtType)

        data = torch.from_numpy(out).float()
        return data

class MTF(object):
    '''
    Use Markov Transition Fields to transfer the time series to images
    '''
    def __init__(self, bins=8, image_size=128):
        self.bins = bins
        self.image_size = image_size

    def mtfImaging(self, data, bins):
        strategy = 'quantile'
        X = data.reshape(1, -1)
        n_samples, n_timestamps = X.shape
        mtf = MarkovTransitionField(image_size=self.image_size, n_bins=bins, strategy=strategy)
        tag_mtf = mtf.fit_transform(X)
        return tag_mtf

    def __call__(self, data1):
        data = copy.deepcopy(data1)

        # initialize output
        data = np.swapaxes(data, 1, 2)
        out  = np.zeros([*data.shape[:2], self.image_size, self.image_size])
        for channel in range(out.shape[0]):
            for fz in range(out.shape[1]):
                out[channel, fz] = self.mtfImaging(data[channel, fz], self.bins)
        data = torch.from_numpy(out).float()
        return data

class gammaFilter(object):
    '''
    low-pass, high-pass, band-pass, keep only gamma band
    '''
    def __init__(self, lowcut=0.16, highcut=40, band=[26,40], fs=160, ntaps=128):
        self.lowcut = lowcut
        self.highcut = highcut
        self.band = band
        self.fs = fs
        self.ntaps = ntaps
    
    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_highpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def bandpass_firwin(self, ntaps, lowcut, highcut, fs, window='hann'):
        nyq = 0.5 * fs
        filter = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False, window=window, scale=False)
        return filter

    def firwin_bandpass_filter(self, data, ntaps, lowcut, highcut, fs, window='hann'):
        filter = self.bandpass_firwin(ntaps, lowcut, highcut, fs, window)
        y = lfilter(filter, 1, data)
        return y

    def __call__(self, data1):
        data = copy.deepcopy(data1)
        out = np.zeros([*data.shape])

        for channel in range(data.shape[0]):
            out[channel,:] = self.butter_lowpass_filter(data[channel,:], self.lowcut, self.fs)
            out[channel,:] = self.butter_highpass_filter(data[channel,:], self.highcut, self.fs)
            out[channel,:] = self.firwin_bandpass_filter(data[channel,:], self.ntaps, self.band[0], self.band[1], self.fs)
        
        data = torch.from_numpy(out).float()
        return data

class MSD(object):
    def __call__(self, data1):
        data = copy.deepcopy(data1).numpy()
        out = np.zeros([data.shape[0]*2, 1])
        for channel in range(data.shape[0]//2):
            out[channel,:] = np.mean(data[channel,:])
            out[channel+data.shape[0],:] = np.std(data[channel,:])
        data = torch.from_numpy(out).float()
        return data
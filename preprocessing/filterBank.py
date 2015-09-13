# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:24:01 2015.

@author: fornax, alexandre
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.signal import butter, lfilter


class FilterBank(BaseEstimator, TransformerMixin):

    """Filterbank TransformerMixin.

    Return signal processed by a bank of butterworth filters.
    """

    def __init__(self, filters='LowpassBank'):
        """init."""
        if filters == 'LowpassBank':
            self.freqs_pairs = [[0.5], [1], [2], [3], [4], [5], [7], [9], [15],
                                [30]]
        else:
            self.freqs_pairs = filters
        self.filters = filters

    def fit(self, X, y=None):
        """Fit Method, Not used."""
        return self

    def transform(self, X, y=None):
        """Transform. Apply filters."""
        X_tot = None
        for freqs in self.freqs_pairs:
            if len(freqs) == 1:
                b, a = butter(5, freqs[0] / 250.0, btype='lowpass')
            else:
                if freqs[1] - freqs[0] < 3:
                    b, a = butter(3, np.array(freqs) / 250.0, btype='bandpass')
                else:
                    b, a = butter(5, np.array(freqs) / 250.0, btype='bandpass')
            X_filtered = lfilter(b, a, X, axis=0)
            X_tot = X_filtered if X_tot is None else np.c_[X_tot, X_filtered]

        return X_tot

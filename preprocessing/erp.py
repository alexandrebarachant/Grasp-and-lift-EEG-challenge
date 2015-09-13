# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 22:00:08 2015.

@author: rc, alexandre
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from mne.io import RawArray
from mne.channels import read_montage
from mne import create_info
from mne import find_events, Epochs
from mne.preprocessing import Xdawn
from mne import compute_raw_data_covariance

from pyriemann.utils.covariance import _lwf
from pyriemann.classification import MDM

from preprocessing.aux import getChannelNames, getEventNames, sliding_window


def toMNE(X, y=None):
    """Tranform array into MNE for epoching."""
    ch_names = getChannelNames()
    montage = read_montage('standard_1005', ch_names)
    ch_type = ['eeg']*len(ch_names)
    data = X.T
    if y is not None:
        y = y.transpose()
        ch_type.extend(['stim']*6)
        event_names = getEventNames()
        ch_names.extend(event_names)
        # concatenate event file and data
        data = np.concatenate((data, y))
    info = create_info(ch_names, sfreq=500.0, ch_types=ch_type,
                       montage=montage)
    raw = RawArray(data, info, verbose=False)
    return raw


def get_epochs_and_cov(X, y, window=500):
    """return epochs from array."""
    raw_train = toMNE(X, y)
    picks = range(len(getChannelNames()))

    events = list()
    events_id = dict()
    for j, eid in enumerate(getEventNames()):
        tmp = find_events(raw_train, stim_channel=eid, verbose=False)
        tmp[:, -1] = j + 1
        events.append(tmp)
        events_id[eid] = j + 1

    # concatenate and sort events
    events = np.concatenate(events, axis=0)
    order_ev = np.argsort(events[:, 0])
    events = events[order_ev]

    epochs = Epochs(raw_train, events, events_id,
                    tmin=-(window / 500.0) + 1 / 500.0 + 0.150,
                    tmax=0.150, proj=False, picks=picks, baseline=None,
                    preload=True, add_eeg_ref=False, verbose=False)

    cov_signal = compute_raw_data_covariance(raw_train, verbose=False)
    return epochs, cov_signal


class ERP(BaseEstimator, TransformerMixin):

    """ERP cov estimator.

    This is a transformer for estimating special form covariance matrices in
    the context of ERP detection [1,2]. For each class, the ERP is estimated by
    averaging all the epochs of a given class. The dimentionality of the ERP is
    reduced using a XDAWN algorithm and then concatenated with the epochs
    before estimation of the covariance matrices.

    References :
    [1] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information
    Geometry", arXiv:1409.0107
    [2] M. Congedo, A. Barachant, A. Andreev ,"A New generation of
    Brain-Computer Interface Based on Riemannian Geometry", arXiv: 1310.8115.
    """

    def __init__(self, window=500, nfilters=3, subsample=1):
        """Init."""
        self.window = window
        self.nfilters = nfilters
        self.subsample = subsample

    def fit(self, X, y):
        """fit."""
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        """fit and return epochs."""
        epochs, cov_signal = get_epochs_and_cov(X, y, self.window)

        xd = Xdawn(n_components=self.nfilters, signal_cov=cov_signal,
                   correct_overlap=False)
        xd.fit(epochs)

        P = []
        for eid in getEventNames():
            P.append(np.dot(xd.filters_[eid][:, 0:self.nfilters].T,
                            xd.evokeds_[eid].data))
        self.P = np.concatenate(P, axis=0)
        self.labels_train = epochs.events[:, -1]
        return epochs

    def transform(self, X, y=None):
        """Transform."""
        test_cov = sliding_window(X.T, window=self.window,
                                  subsample=self.subsample,
                                  estimator=self.erp_cov)
        return test_cov

    def fit_transform(self, X, y):
        """Fit and transform."""
        epochs = self._fit(X, y)
        train_cov = np.array([self.erp_cov(ep) for ep in epochs.get_data()])
        return train_cov

    def erp_cov(self, X):
        """Compute ERP covariances."""
        data = np.concatenate((self.P, X), axis=0)
        return _lwf(data)

    def update_subsample(self, old_sub, new_sub):
        """update subsampling."""
        self.subsample = new_sub


class ERPDistance(BaseEstimator, TransformerMixin):

    """ERP distance cov estimator.

    This transformer estimates Riemannian distance for ERP covariance matrices.
    After estimation of special form ERP covariance matrices using the ERP
    transformer, a MDM [1] algorithm is used to compute Riemannian distance.

    References:
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012
    """

    def __init__(self, window=500, nfilters=3, subsample=1, metric='riemann',
                 n_jobs=1):
        """Init."""
        self.window = window
        self.nfilters = nfilters
        self.subsample = subsample
        self.metric = metric
        self.n_jobs = n_jobs
        self._fitted = False

    def fit(self, X, y):
        """fit."""
        # Create ERP and get cov mat
        self.ERP = ERP(self.window, self.nfilters, self.subsample)
        train_cov = self.ERP.fit_transform(X, y)
        labels_train = self.ERP.labels_train

        # Add rest epochs
        rest_cov = self._get_rest_cov(X, y)
        train_cov = np.concatenate((train_cov, rest_cov), axis=0)
        labels_train = np.concatenate((labels_train, [0] * len(rest_cov)))

        # fit MDM
        self.MDM = MDM(metric=self.metric, n_jobs=self.n_jobs)
        self.MDM.fit(train_cov, labels_train)
        self._fitted = True
        return self

    def transform(self, X, y=None):
        """Transform."""
        test_cov = self.ERP.transform(X)
        dist = self.MDM.transform(test_cov)
        dist = dist[:, 1:] - np.atleast_2d(dist[:, 0]).T
        return dist

    def update_subsample(self, old_sub, new_sub):
        """update subsampling."""
        if self._fitted:
            self.ERP.update_subsample(old_sub, new_sub)

    def _get_rest_cov(self, X, y):
        """Sample rest epochs from data and compute the cov mat."""
        ix = np.where(np.diff(y[:, 0]) == 1)[0]
        rest = []
        offset = - self.window
        for i in ix:
            start = i + offset - self.window
            stop = i + offset
            rest.append(self.ERP.erp_cov(X[slice(start, stop)].T))
        return np.array(rest)

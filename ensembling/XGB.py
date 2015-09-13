# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 21:19:51 2015.

@author: rc, alex
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker

from preprocessing.aux import delay_preds
import xgboost as xgb


class XGB(BaseEstimator, ClassifierMixin):

    """Ensembling with eXtreme Gradient Boosting."""

    def __init__(self, ensemble, n_estimators=100, max_depth=5, subsample=0.7,
                 nthread=12,delay=None,skip=None,subsample_data=1,partsTest=1, jump=None):
        """Init."""
        self.ensemble = ensemble
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.nthread = nthread
        
        ### timecourse history parameters ###
        # how many past time samples to include along with the most recent sample
        self.applyPreds = delay is not None and skip is not None    
        # how many past time samples to include along with the most recent sample  
        self.delay = delay
        # subsample above samples
        self.skip = skip
        # here can be set a custom subsampling scheme, it overrides previous params
        self.jump = jump
        
        # due to RAM limitations testing data has to be split into 'partsTest' parts
        self.partsTest = partsTest
        
        # subsampling input data as an efficient form of regularization
        self.subsample_data = subsample_data
        
        # used in bagging to set different starting points when subsampling the data
        self.mdlNr = 0

        self.clf = []

    def fit(self, X, y):
        """Fit."""
        
        X = X[(self.mdlNr*5 % self.subsample_data)::self.subsample_data]
        y = y[(self.mdlNr*5 % self.subsample_data)::self.subsample_data]
        
        if self.applyPreds:
            if self.jump is not None:
                X = delay_preds(X, delay=self.delay/self.subsample_data, skip=self.skip/self.subsample_data, jump=self.jump/self.subsample_data)
            else:
                X = delay_preds(X, delay=self.delay/self.subsample_data, skip=self.skip/self.subsample_data)
        self.clf = []
        
        widgets = ['Training : ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=6)
        pbar.start()
        
        # training separate models for each event
        for col in range(6):
            self.clf.append(xgb.XGBClassifier(n_estimators=self.n_estimators,
                                              max_depth=self.max_depth,
                                              subsample=self.subsample,
                                              nthread=self.nthread))
            self.clf[col].fit(X, y[:, col])
            pbar.update(col)

    def _predict_proba(self,X):
        """Predict probability for each event separately, then concatenate results."""
        pred = []
        for col in range(6):
            pred.append(self.clf[col].predict_proba(X)[:, 1])
        pred = np.vstack(pred).transpose()
        return pred
        
    def predict_proba(self, X):
        """Predict probability."""
        if self.applyPreds:
            p = np.zeros((X.shape[0],6))
            for part in range(self.partsTest):
                start = part*X.shape[0]//self.partsTest-self.delay*(part>0)
                stop = (part+1)*X.shape[0]//self.partsTest
                X_delayed = delay_preds(X[slice(start,stop)], delay=self.delay, skip=self.skip, jump=self.jump)[self.delay*(part>0):]
                start += self.delay*(part>0)
                p[slice(start,stop)] += self._predict_proba(X_delayed)
                X_delayed = None
            return p
        else:
            return self._predict_proba(X)
        

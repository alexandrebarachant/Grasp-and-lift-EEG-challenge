# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:12:12 2015.

@author: rc, alex
"""
import numpy as np
from collections import OrderedDict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp

from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker


class WeightedMeanClassifier(BaseEstimator, ClassifierMixin):
    
    """Weigted mean classifier with AUC optimization."""
    
    def __init__(self, ensemble, step=0.025, max_evals=100, mean='arithmetic', 
                 verbose=True):
        """Init."""
        self.ensemble = ensemble
        self.step = step
        self.max_evals = max_evals
        self.mean = mean
        self.count = -1
        self.verbose = verbose
        
        self.param_space = OrderedDict()
        for model in ensemble:
            self.param_space[model] = hp.quniform(model, 0, 3, self.step)
            
        # input data are arranged in a particular order, whereas hyperopt uses 
        # unordered lists when optimizing. The model has to keep track 
        # of the initial order so that correct weights are applied to columns
        self.sorting = dict()
        for i, m in enumerate(self.ensemble):
            self.sorting[m] = i
    
    def fit(self, X, y):
        """Fit."""
        self.best_params = None
        if self.mean != 'simple':
            if self.verbose:
                widgets = ['Training : ', Percentage(), ' ', Bar(marker=RotatingMarker()),
                   ' ', ETA(), ' ']
                self.pbar = ProgressBar(widgets=widgets, maxval=(self.max_evals * len(self.param_space)))
                self.pbar.start()
            
            objective = lambda w: -np.mean([roc_auc_score(y[:, col],
                                            self.calcMean(X[:, col::6], w, training=True))
                                            for col in range(6)])
                                            
            self.best_params = fmin(objective, self.param_space, algo=tpe.suggest,
                                    max_evals=self.max_evals)
            
            if self.verbose:
                print(self.best_params)
        else:
            self.best_params = None
    
    def predict_proba(self, X):
        """Get predictions."""
        return np.c_[[self.calcMean(X[:, col::6], self.best_params)
                      for col in range(6)]].transpose()
    
    def calcMean(self, X, w, training = False):
        """Calculate Mean according to weights."""
        self.count += 1
        if self.verbose and self.count <= (self.max_evals * len(self.param_space)) and not self.count%10 and training:
            self.pbar.update(self.count)
        
        if self.mean == 'simple':
            return np.sum(X, axis=1)/X.shape[1]
        else:
            w = [w[k] for k in sorted(self.sorting, key=self.sorting.get)]
            if self.mean == 'arithmetic':
                return np.sum(X * w, axis=1)/np.sum(w)
            elif self.mean == 'geometric':
                return np.exp(np.sum(np.log(X) * w, axis=1)/np.sum(w))
            elif self.mean == 'power':
                return 1/(1+np.exp(-np.sum(X ** w, axis=1)))
            else:
                print 'Mean should be either "simple", "arithmetic", "geometric" or "power"'

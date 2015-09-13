# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:12:12 2015

@author: rc, alex
"""
import os
import sys
if __name__ == '__main__' and __package__ is None:
    filePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(filePath)

import pandas as pd
import numpy as np
import yaml
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import LeaveOneLabelOut

from preprocessing.aux import getEventNames
from utils.ensembles import createEnsFunc, loadPredictions

from ensembling.WeightedMean import WeightedMeanClassifier

yml = yaml.load(open(sys.argv[1]))

fileName = yml['fileName']
ensemble = yml['ensemble']
subsample = yml['subsample'] if 'subsample' in yml else 1
seed = yml['seed'] if 'seed' in yml else 4234521 
mean_type = yml['mean_type'] if 'mean_type' in yml else 'arithmetic'
verbose = yml['verbose'] if 'verbose' in yml else True
print mean_type
print ensemble
np.random.seed(seed)
print 'Running weighted mean ensemble, results will be saved in submissions/%s.csv' % fileName

models = []
for m in mean_type:
    models.append(WeightedMeanClassifier(ensemble, mean=m, verbose=verbose))

######
cols = getEventNames()

ids = np.load('../infos_test.npy')
subjects_test = ids[:, 1]
series_test = ids[:, 2]
ids = ids[:, 0]
labels = np.load('../infos_val.npy')
subjects = labels[:, -2]
series = labels[:, -1]
labels = labels[:, :-2]

allCols = range(len(cols))

# ## loading prediction ###
files = ensemble
preds_val = OrderedDict()
for f in files:
    loadPredictions(preds_val, f, [f], lvl=2)

# ## train/test ###
aggr = createEnsFunc(ensemble)
dataTrain = aggr(preds_val)
preds_val = None

# do CV
aucs = []
cv = LeaveOneLabelOut(series)
p = np.zeros(labels.shape)
for train,test in cv:
    currentSeries = np.unique(series[test])[0]
    for m in range(len(models)):
        models[m].fit(dataTrain[train][::subsample], labels[train][::subsample])
        p[test] += models[m].predict_proba(dataTrain[test]) / len(mean_type)
    aucs.append(np.mean([roc_auc_score(labels[test],p[test])]))
    print 'score on series %d: %.5f' % (currentSeries, aucs[-1])
print 'CV score: %.5f / %.6f' % (np.mean(aucs), np.std(aucs))
np.save('val/val_%s.npy'%fileName,[p])

# train WMs on all training data
models = []
for m in mean_type:
    wm = WeightedMeanClassifier(ensemble, mean=m, verbose=verbose)
    wm.fit(dataTrain[::subsample], labels[::subsample])
    models.append(wm)

dataTrain = None

# load test data
preds_test = OrderedDict()
for f in files:
    loadPredictions(preds_test, f, [f], lvl=2, test=True)
dataTest = aggr(preds_test)
preds_test = None

# get predictions
p = 0
for m in range(len(models)):
    p += models[m].predict_proba(dataTest) / len(models)

# generate submission
sub = pd.DataFrame(data=p,index=ids,columns=cols)
sub.to_csv('submissions/%s.csv'%fileName,index_label='id',float_format='%.8f')
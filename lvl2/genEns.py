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

import numpy as np
import yaml
from copy import deepcopy
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import LeaveOneLabelOut

from preprocessing.aux import getEventNames
from utils.ensembles import createEnsFunc, loadPredictions, getLvl1ModelList

from ensembling.WeightedMean import WeightedMeanClassifier
from ensembling.NeuralNet import NeuralNet
from ensembling.XGB import XGB


def _from_yaml_to_func(method, params):
    """go from yaml to method.

    Need to be here for accesing local variables.
    """
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)

# ## here read YAML and build models ###
yml = yaml.load(open(sys.argv[1]))

fileName = yml['Meta']['file']
if 'subsample' in yml['Meta']:
    subsample = yml['Meta']['subsample']
else:
    subsample = 1

modelName, modelParams = yml['Model'].iteritems().next()
model_base = _from_yaml_to_func(modelName, modelParams)

ensemble = yml['Model'][modelName]['ensemble']
addSubjectID = True if 'addSubjectID' in yml.keys() else False

mode = sys.argv[2]
if mode == 'val':
    test = False
elif mode == 'test':
    test = True
else:
    raise('Invalid mode. Please specify either val or test')

print('Running %s in mode %s, predictions will be saved as %s' % (modelName,mode,fileName))

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

# ## loading predictions ###
files = getLvl1ModelList()

preds_val = OrderedDict()
for f in files:
    loadPredictions(preds_val, f[0], f[1])
# validity check
for m in ensemble:
    assert(m in preds_val)

# ## train/test ###
aggr = createEnsFunc(ensemble)
dataTrain = aggr(preds_val)
preds_val = None

# optionally adding subjectIDs
if addSubjectID:
    dataTrain = np.c_[dataTrain, subjects]

np.random.seed(4234521)

if test:
    # train the model
    model = deepcopy(model_base)
    model.fit(dataTrain[::subsample], labels[::subsample])
    dataTrain = None

    # load test data
    preds_test = OrderedDict()
    for f in files:
        loadPredictions(preds_test, f[0], f[1], test=True)
    dataTest = aggr(preds_test)
    preds_test = None
    # switch to add subjects
    if addSubjectID:
        dataTest = np.c_[dataTest, subjects_test]

    # get predictions
    p = model.predict_proba(dataTest)
    np.save('test/test_%s.npy' % fileName, [p])
else:
    auc_tot = []
    p = np.zeros(labels.shape)
    cv = LeaveOneLabelOut(series)
    for fold, (train, test) in enumerate(cv):
        model = deepcopy(model_base)
        if modelName == 'NeuralNet':
            # passing also test data to print out test error during training
            model.fit(dataTrain[train], labels[train], dataTrain[test],
                      labels[test])
        else:
            model.fit(dataTrain[train][::subsample], labels[train][::subsample])
        p[test] = model.predict_proba(dataTrain[test])
        auc = [roc_auc_score(labels[test][:, col], p[test][:, col])
               for col in allCols]
        auc_tot.append(np.mean(auc))
        print('Fold %d, score: %.5f' % (fold, auc_tot[-1]))
    print('AUC: %.5f' % np.mean(auc_tot))
    np.save('val/val_%s.npy' % fileName, [p])

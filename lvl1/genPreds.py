# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:56:55 2015.

@author: alex, rc

This script contain code to generate lvl1 model prediction.
usage : python genPreds.py model_name mode
with mode = val for validation and val = test for test.

This script will read the model description from the yaml file, load
dependencies, create preprocessing and classification pipeline and apply them
on raw data independently on each subjects.

This script support caching of preprocessed data, in order to allow reuse of
preprocessing pipeline across model.
"""
import os
import sys
if __name__ == '__main__' and __package__ is None:
    filePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(filePath)

import numpy as np
import pandas as pd
from time import time
from copy import deepcopy
import yaml
from sklearn.pipeline import make_pipeline, Pipeline
from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker

from sklearn.metrics import roc_auc_score

from preprocessing.aux import getEventNames, load_raw_data

from multiprocessing import Pool
cols = getEventNames()


def _from_yaml_to_func(method, params):
    """go from yaml to method.

    Need to be here for accesing local variables.
    """
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)


def doCols(col):
    """Train and Predict for one event."""
    p = []
    for clf in clfs:
        clf.fit(trainPreprocessed, labels_train[:, col])
        p.append(clf.predict_proba(testPreprocessed)[:, 1])
    return p


yml = yaml.load(open(sys.argv[1]))

# Import package
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# meta settings
fileName = yml['Meta']['file']
cores = yml['Meta']['cores']
subsample = yml['Meta']['subsample']
cache_preprocessed = yml['Meta']['cachePreprocessed']

if 'subsample_test' in yml['Meta'].keys():
    subsample_test = yml['Meta']['subsample_test']
else:
    subsample_test = 1

if 'addPreprocessed' in yml['Meta']:
    addPreprocessed = yml['Meta']['addPreprocessed']
else:
    addPreprocessed = []

# preprocessing pipeline
pipe = []
for item in yml['Preprocessing']:
    for method, params in item.iteritems():
        pipe.append(_from_yaml_to_func(method, params))
preprocess_base = make_pipeline(*pipe)

# post preprocessing
postpreprocess_base = None
if 'PostPreprocessing' in yml.keys():
    pipe = []
    for item in yml['PostPreprocessing']:
        for method, params in item.iteritems():
            pipe.append(_from_yaml_to_func(method, params))
    postpreprocess_base = make_pipeline(*pipe)

# models
clfs = []
for mdl in yml['Models']:
    clfs.append('Pipeline([ %s ])' % mdl)

for i, clf in enumerate(clfs):
    clfs[i] = eval(clf)

# ## read arguments ###

mode = sys.argv[2]
if mode == 'val':
    test = False
elif mode == 'test':
    test = True
else:
    raise('Invalid mode. Please specify either val or test')

if test:
    folder = 'test/'
    prefix = 'test_'
else:
    folder = 'val/'
    prefix = 'val_'


print 'Running %s, to be saved in file %s' % (mode, fileName)

saveFolder = folder + fileName
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

# #### define lists #####
subjects = range(1, 13)
widgets = ['Cross Val : ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA(), ' ']
pbar = ProgressBar(widgets=widgets, maxval=len(subjects))
pbar.start()

report = pd.DataFrame(index=[fileName])
start_time = time()
# #### generate predictions #####
for subject in subjects:
    print 'Loading data for subject %d...' % subject
    # ############### READ DATA ###############################################
    data_train, labels_train, data_test, labels_test = load_raw_data(subject,
                                                                     test)

    trainPreprocessed = None
    testPreprocessed = None
    cacheFile = '%s/train_sub%d.npy' % (saveFolder, subject)
    # copy processing pipeline to start fresh
    preprocess = deepcopy(preprocess_base)
    if postpreprocess_base is not None:
        postpreprocess = deepcopy(postpreprocess_base)
    else:
        postpreprocess = None

    # ### preprocessing ####
    print 'Preprocessing Training data...'

    if cache_preprocessed and os.path.isfile(cacheFile):
        # if cache activated + file exist, load file
        trainPreprocessed = np.load(cacheFile)
    else:
        # if not, do preprocessing
        trainPreprocessed = preprocess.fit_transform(data_train, labels_train)
        # if cache activated but no file, save
        if cache_preprocessed:
            np.save(cacheFile, trainPreprocessed)
            trainPreprocessed = None
    data_train = None

    print 'Preprocessing Test data...'
    cacheFile = '%s/test_sub%d.npy' % (saveFolder, subject)

    # update subsampling for test Preprocessing
    for name, step in preprocess.steps:
        if hasattr(step, 'update_subsample'):
            step.update_subsample(subsample, subsample_test)

    if cache_preprocessed and os.path.isfile(cacheFile):
        # if cache activated + file exist, load file
        testPreprocessed = np.load(cacheFile)
    else:
        # if not, do preprocessing
        testPreprocessed = preprocess.transform(data_test)
        # if cache activated but no file, save
        if cache_preprocessed:
            np.save(cacheFile, testPreprocessed)
    data_test = None

    print 'Post Preprocessing data...'
    if cache_preprocessed and (trainPreprocessed is None):
        # if cache activated load file
        cacheFile = '%s/train_sub%d.npy' % (saveFolder, subject)
        trainPreprocessed = np.load(cacheFile)

    # Add preprocessed feature if they have been set in the config file
    for feat_name in addPreprocessed:
        featFile = '%s/%s/train_sub%d.npy' % (folder, feat_name, subject)
        if os.path.isfile(featFile):
            feat = np.load(featFile)
            if trainPreprocessed is None:
                trainPreprocessed = feat
            else:
                trainPreprocessed = np.c_[trainPreprocessed, feat]
            feat = None
        else:
            raise ValueError("File %s does not exist" % featFile)

    # Add preprocessed feature if they have been set in the config file
    for feat_name in addPreprocessed:
        featFile = '%s/%s/test_sub%d.npy' % (folder, feat_name, subject)
        if os.path.isfile(featFile):
            feat = np.load(featFile)
            if testPreprocessed is None:
                testPreprocessed = feat
            else:
                testPreprocessed = np.c_[testPreprocessed, feat]
            feat = None
        else:
            raise ValueError('File %s does not exist' % featFile)

    trainPreprocessed[np.isnan(trainPreprocessed)] = 0
    testPreprocessed[np.isnan(testPreprocessed)] = 0

    if postpreprocess is not None:
        trainPreprocessed = postpreprocess.fit_transform(trainPreprocessed,
                                                         labels_train)
        for name, step in postpreprocess.steps:
            if hasattr(step, 'update_subsample'):
                step.update_subsample(subsample, subsample_test)

        testPreprocessed = postpreprocess.transform(testPreprocessed)

    print 'Training models...'
    labels_train = labels_train[::subsample]
    if cores == 1:
        preds = [doCols(i) for i in range(len(cols))]
    else:
        pool = Pool(processes=cores)
        preds = pool.map(doCols, range(len(cols)))
        pool.close()
    # ### results #####
    print 'Aggregating results...'
    for i in range(len(clfs)):
        pred_i = [j[i] for j in preds]
        pred_i = np.array(np.vstack(pred_i)).transpose()
        np.save('%s/sub%d_clf%d.npy' % (saveFolder, subject, i), pred_i)
        if not test:
            auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in
                          zip(labels_test[::subsample_test].T, pred_i.T)])
            print '%d, clf %d: %.5f' % (subject, i, auc)

    # clear memory
    preds = None
    trainPreprocessed = None
    testPreprocessed = None

    # update progress Bar
    pbar.update(subject)


if not test:
    labels = np.load('../infos_val.npy')[:, :-2]

# ## AGGREGATE HERE
preds_tot = []

for i in range(len(clfs)):
    preds_tot.append([])
    for subject in subjects:
        preds_tot[i].append(np.load('%s/sub%d_clf%d.npy' % (saveFolder,
                                                            subject, i)))
    preds_tot[i] = np.concatenate(preds_tot[i])
    if not test:
        auc = [roc_auc_score(trueVals, p) for trueVals, p in
               zip(labels[::subsample_test].T, preds_tot[i].T)]
        report['AUC'] = np.mean(auc)
        print np.mean(auc)

# ## save the model ###
np.save(folder + prefix + fileName + '.npy', preds_tot)

# ## save report
end_time = time()
report['Time'] = end_time - start_time
report.to_csv("report/%s_%s.csv" % (prefix, fileName))
print report

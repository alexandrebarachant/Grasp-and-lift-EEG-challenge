# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:56:55 2015.

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
from time import time
from copy import deepcopy
from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from preprocessing.aux import load_raw_data
from ensembling.NeuralNet import NeuralNet


def _from_yaml_to_func(method, params):
    """go from yaml to method.

    Need to be here for accesing local variables.
    """
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)

# ## read model parameters ###
yml = yaml.load(open(sys.argv[1]))

# Import package
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

fileName = yml['Meta']['file']
training_params = yml['Training']
architecture = yml['Architecture']

delay = training_params['delay']
skip = training_params['skip']
parts_train = training_params['parts_train']
parts_test = training_params['parts_test']
smallEpochs = training_params['smallEpochs']
majorEpochs = training_params['majorEpochs']
checkEveryEpochs = training_params['checkEveryEpochs']
subsample = training_params['subsample']

# meta settings
cache_preprocessed = yml['Meta']['cachePreprocessed']

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


# required transformers

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

np.random.seed(4234521)
# #### generate predictions #####
for subject in subjects:
    print 'Loading data for subject %d...' % subject
    # ############### READ DATA ###############################################
    data_train, labels_train, data_test, labels_test = load_raw_data(subject,
                                                                     test)
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

    if postpreprocess is not None:
        trainPreprocessed = postpreprocess.fit_transform(trainPreprocessed,
                                                         labels_train)

    trainPreprocessed[np.isnan(trainPreprocessed)] = 0
    # update subsampling for test Preprocessing
    for name, step in preprocess.steps:
        if hasattr(step, 'update_subsample'):
            step.update_subsample(subsample, 1)

    if postpreprocess is not None:
        for name, step in postpreprocess.steps:
            if hasattr(step, 'update_subsample'):
                step.update_subsample(subsample, 1)

    print 'Preprocessing Test data...'
    cacheFile = '%s/test_sub%d.npy' % (saveFolder, subject)

    if cache_preprocessed and os.path.isfile(cacheFile):
        # if cache activated + file exist, load file
        testPreprocessed = np.load(cacheFile)
    else:
        # if not, do preprocessing
        testPreprocessed = preprocess.transform(data_test)
        # if cache activated but no file, save
        if cache_preprocessed:
            np.save(cacheFile, testPreprocessed)

    if postpreprocess is not None:
        testPreprocessed = postpreprocess.transform(testPreprocessed)
    testPreprocessed[np.isnan(testPreprocessed)] = 0

    model = NeuralNet(None, architecture, training_params, 
                      partsTrain=parts_train,partsTest=parts_test,
                      delay=delay,skip=skip,subsample=subsample,
                      majorEpochs=majorEpochs,smallEpochs=smallEpochs,
                      checkEveryEpochs=checkEveryEpochs)

    model.fit(trainPreprocessed,labels_train,testPreprocessed,labels_test)
    
    preds = model.predict_proba(testPreprocessed)

    if not test:
        auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in
                      zip(labels_test.T, preds.T)])
        print("%d, test AUC : %.5f" % (subject, auc))

    np.save('%s/sub%d.npy' % (saveFolder, subject), preds)

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
for subject in subjects:
    preds_tot.append(np.load('%s/sub%d.npy' % (saveFolder, subject)))
preds_tot = np.concatenate(preds_tot)
if not test:
    auc = [roc_auc_score(trueVals, p) for trueVals, p in
                  zip(labels.transpose(), preds_tot.transpose())]
    print np.mean(auc)
    report['AUC'] = np.mean(auc)
preds_tot = [preds_tot]

# ## save the model ###
np.save(folder + prefix + fileName + '.npy', preds_tot)
end_time = time()
report['Time'] = end_time - start_time
report.to_csv("report/%s_%s.csv" % (prefix, fileName))
print report
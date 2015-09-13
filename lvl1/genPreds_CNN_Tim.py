# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 21:28:32 2015.

Script written by Tim Hochberg with parameter tweaks by Bluefool.
https://www.kaggle.com/bitsofbits/grasp-and-lift-eeg-detection/naive-nnet

Modifications: rc, alex
"""
import os
import sys
if __name__ == '__main__' and __package__ is None:
    filePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(filePath)

import yaml
from glob import glob
import numpy as np
import pandas as pd
from time import time

from sklearn.metrics import roc_auc_score

# Lasagne (& friends) imports
import theano
from nolearn.lasagne import BatchIterator, NeuralNet, TrainSplit
from lasagne.objectives import aggregate, binary_crossentropy
from lasagne.layers import (InputLayer, DropoutLayer, DenseLayer, Conv1DLayer,
                            Conv2DLayer)
from lasagne.updates import nesterov_momentum, adam
from theano.tensor.nnet import sigmoid

from mne import concatenate_raws, pick_types

from preprocessing.aux import creat_mne_raw_object
from preprocessing.filterBank import FilterBank

# Silence some warnings from lasagne
import warnings
warnings.filterwarnings('ignore', '.*topo.*')
warnings.filterwarnings('ignore', module='.*lasagne.init.*')
warnings.filterwarnings('ignore', module='.*nolearn.lasagne.*')

####
yml = yaml.load(open(sys.argv[1]))
fileName = yml['Meta']['file']

filt2Dsize = yml['filt2Dsize'] if 'filt2Dsize' in yml.keys() else 0
filters = yml['filters']
delay = yml['delay']
skip = yml['skip']

if 'bags' in yml.keys():
    bags = yml['bags']
else:
    bags = 3

mode = sys.argv[2]
if mode == 'val':
    test = False
elif mode == 'test':
    test = True
else:
    raise('Invalid mode. Please specify either val or test')


###########
SUBJECTS = list(range(1, 13))
TRAIN_SERIES = list(range(1, 9))
TEST_SERIES = [9, 10]

N_ELECTRODES = 32
N_EVENTS = 6

SAMPLE_SIZE = delay
DOWNSAMPLE = skip
TIME_POINTS = SAMPLE_SIZE // DOWNSAMPLE

TRAIN_SIZE = 5120

# We encapsulate the event / electrode data in a Source object.


def preprocessData(data):
    """Preprocess data with filterbank."""
    fb = FilterBank(filters)
    return fb.transform(data)


class Source:

    """Loads, preprocesses and holds data."""

    mean = None
    std = None

    def load_raw_data(self, subject, series):
        """Load data for a subject / series."""
        test = series == TEST_SERIES
        if not test:
            fnames = [glob('../data/train/subj%d_series%d_data.csv' %
                      (subject, i)) for i in series]
        else:
            fnames = [glob('../data/test/subj%d_series%d_data.csv' %
                      (subject, i)) for i in series]
        fnames = list(np.concatenate(fnames))
        fnames.sort()
        raw_train = [creat_mne_raw_object(fname, read_events=not test)
                     for fname in fnames]
        raw_train = concatenate_raws(raw_train)
        # pick eeg signal
        picks = pick_types(raw_train.info, eeg=True)

        self.data = raw_train._data[picks].transpose()

        self.data = preprocessData(self.data)

        if not test:

            self.events = raw_train._data[32:].transpose()

    def normalize(self):
        """normalize data."""
        self.data -= self.mean
        self.data /= self.std


class TrainSource(Source):

    """Source for training data."""

    def __init__(self, subject, series_list):
        """Init."""
        self.load_raw_data(subject, series_list)
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.normalize()


# Note that Test/Submit sources use the mean/std from the training data.
# This is both standard practice and avoids using future data in theano
# test set.

class TestSource(Source):

    """Source for test data."""

    def __init__(self, subject, series, train_source):
        """Init."""
        self.load_raw_data(subject, series)
        self.mean = train_source.mean
        self.std = train_source.std
        self.normalize()


# Lay out the Neural net.


class LayerFactory:

    """Helper class that makes laying out Lasagne layers more pleasant."""

    def __init__(self):
        """Init."""
        self.layer_cnt = 0
        self.kwargs = {}

    def __call__(self, layer, layer_name=None, **kwargs):
        """Call."""
        self.layer_cnt += 1
        name = layer_name or "layer{0}".format(self.layer_cnt)
        for k, v in kwargs.items():
            self.kwargs["{0}_{1}".format(name, k)] = v
        return (name, layer)


class IndexBatchIterator(BatchIterator):

    """Generate BatchData from indices.

    Rather than passing the data into the fit function, instead we just pass in
    indices to the data.  The actual data is then grabbed from a Source object
    that is passed in at the creation of the IndexBatchIterator. Passing in a
    '-1' grabs a random value from the Source.

    As a result, an "epoch" here isn't a traditional epoch, which looks at all
    the time points. Instead a random subsamle of 0.8*TRAIN_SIZE points from
    the training data are used each "epoch" and 0.2 TRAIN_SIZE points are use
    for validation.

    """

    def __init__(self, source, *args, **kwargs):
        """Init."""
        super(IndexBatchIterator, self).__init__(*args, **kwargs)
        self.source = source
        if source is not None:
            # Tack on (SAMPLE_SIZE-1) copies of the first value so that it is
            # easy to grab
            # SAMPLE_SIZE POINTS even from the first location.
            x = source.data
            input_shape = [len(x) + (SAMPLE_SIZE - 1), N_ELECTRODES]
            self.augmented = np.zeros(input_shape, dtype=np.float32)
            self.augmented[SAMPLE_SIZE-1:] = x
            self.augmented[:SAMPLE_SIZE-1] = x[0]
        if filt2Dsize:
            input_shape = [self.batch_size, 1, N_ELECTRODES, TIME_POINTS]
            self.Xbuf = np.zeros(input_shape, np.float32)
        else:
            input_shape = [self.batch_size, N_ELECTRODES, TIME_POINTS]
            self.Xbuf = np.zeros(input_shape, np.float32)
        self.Ybuf = np.zeros([self.batch_size, N_EVENTS], np.float32)

    def transform(self, X_indices, y_indices):
        """Transform."""
        X_indices, y_indices = super(IndexBatchIterator,
                                     self).transform(X_indices, y_indices)
        [count] = X_indices.shape
        # Use preallocated space
        X = self.Xbuf[:count]
        Y = self.Ybuf[:count]
        for i, ndx in enumerate(X_indices):
            if ndx == -1:
                ndx = np.random.randint(len(self.source.events))
            sample = self.augmented[ndx:ndx+SAMPLE_SIZE]
            # Reverse so we get most recent point, otherwise downsampling drops
            # the last
            # DOWNSAMPLE-1 points.
            if filt2Dsize:
                X[i][0] = sample[::-1][::DOWNSAMPLE].transpose()
            else:
                X[i] = sample[::-1][::DOWNSAMPLE].transpose()

            if y_indices is not None:
                Y[i] = self.source.events[ndx]
        Y = None if (y_indices is None) else Y
        return X, Y


# Simple / Naive net. Borrows from Daniel Nouri's Facial Keypoint Detection
# Tutorial

def create_net(train_source, test_source, batch_size=128, max_epochs=100,
               train_val_split=False):
    """Create NN."""
    if train_val_split:
        train_val_split = TrainSplit(eval_size=0.2)
    else:
        train_val_split = TrainSplit(eval_size=False)

    batch_iter_train = IndexBatchIterator(train_source, batch_size=batch_size)
    batch_iter_test = IndexBatchIterator(test_source, batch_size=batch_size)
    LF = LayerFactory()

    dense = 1024  # larger (1024 perhaps) would be better

    if filt2Dsize:
        inputLayer = LF(InputLayer, shape=(None, 1, N_ELECTRODES, TIME_POINTS))
        convLayer = LF(Conv2DLayer, num_filters=8, filter_size=(N_ELECTRODES, filt2Dsize))
    else:
        inputLayer = LF(InputLayer, shape=(None, N_ELECTRODES, TIME_POINTS))
        convLayer = LF(Conv1DLayer, num_filters=8, filter_size=1)

    layers = [
        inputLayer,
        LF(DropoutLayer, p=0.5),
        convLayer,
        # Standard fully connected net from now on
        LF(DenseLayer, num_units=dense),
        LF(DropoutLayer, p=0.5),
        LF(DenseLayer, num_units=dense),
        LF(DropoutLayer, p=0.5),
        LF(DenseLayer, layer_name="output", num_units=N_EVENTS,
           nonlinearity=sigmoid)
    ]

    def loss(x, t):
        return aggregate(binary_crossentropy(x, t))

    if filt2Dsize:
        nnet = NeuralNet(y_tensor_type=theano.tensor.matrix,
                         layers=layers,
                         batch_iterator_train=batch_iter_train,
                         batch_iterator_test=batch_iter_test,
                         max_epochs=max_epochs,
                         verbose=0,
                         update=adam,
                         update_learning_rate=0.001,
                         objective_loss_function=loss,
                         regression=True,
                         train_split=train_val_split,
                         **LF.kwargs)
    else:
        nnet = NeuralNet(y_tensor_type=theano.tensor.matrix,
                         layers=layers,
                         batch_iterator_train=batch_iter_train,
                         batch_iterator_test=batch_iter_test,
                         max_epochs=max_epochs,
                         verbose=0,
                         update=nesterov_momentum,
                         update_learning_rate=0.02,
                         update_momentum=0.9,
                         # update=adam,
                         # update_learning_rate=0.001,
                         objective_loss_function=loss,
                         regression=True,
                         train_split=train_val_split,
                         **LF.kwargs)

    return nnet


# Do the training.
print 'Running in mode %s, saving to file %s' % (mode,fileName)
report = pd.DataFrame(index=[fileName])
start_time = time()

train_indices = np.zeros([TRAIN_SIZE], dtype=int) - 1

np.random.seed(67534)

valid_series = [7, 8]
max_epochs = 100

if test is False:
    probs_bags = []
    for bag in range(bags):
        probs_tot = []
        lbls_tot = []
        for subject in range(1, 13):
            tseries = sorted(set(TRAIN_SERIES) - set(valid_series))
            train_source = TrainSource(subject, tseries)
            test_source = TestSource(subject, valid_series, train_source)
            net = create_net(train_source, test_source, max_epochs=max_epochs,
                             train_val_split=False)
            dummy = net.fit(train_indices, train_indices)
            indices = np.arange(len(test_source.data))
            probs = net.predict_proba(indices)
    
            auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in
                          zip(test_source.events.T, probs.T)])
            print 'Bag %d, subject %d, AUC: %.5f' % (bag, subject, auc)
            probs_tot.append(probs)
            lbls_tot.append(test_source.events)
    
        probs_tot = np.concatenate(probs_tot)
        lbls_tot = np.concatenate(lbls_tot)
        auc = np.mean([roc_auc_score(trueVals, p) for trueVals, p in
                      zip(lbls_tot.transpose(), probs_tot.transpose())])
        print auc
        probs_bags.append(probs_tot)
    
    probs_bags = np.mean(probs_bags, axis=0)
    np.save('val/val_%s.npy' % fileName, [probs_bags])

else:
    probs_bags = []
    for bag in range(bags):
        probs_tot = []
        for subject in range(1, 13):
            tseries = set(TRAIN_SERIES)
            train_source = TrainSource(subject, tseries)
            test_source = TestSource(subject, TEST_SERIES, train_source)
            net = create_net(train_source, test_source, max_epochs=max_epochs,
                             train_val_split=False)
            dummy = net.fit(train_indices, train_indices)
            indices = np.arange(len(test_source.data))
            probs = net.predict_proba(indices)
            print 'Bag %d, subject %d' % (bag, subject)
            probs_tot.append(probs)
        probs_tot = np.concatenate(probs_tot)
        probs_bags.append(probs_tot)
    
    probs_bags = np.mean(probs_bags, axis=0)
    np.save('test/test_%s.npy' % fileName, [probs_bags])


prefix = 'test_' if test else 'val_'
end_time = time()
report['Time'] = end_time - start_time
report.to_csv("report/%s_%s.csv" % (prefix, fileName))
print report

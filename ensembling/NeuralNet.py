# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 18:18:11 2015

@author: rc, alex
"""

import numpy as np
from sklearn.base  import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from progressbar import Bar, ETA, Percentage, ProgressBar, RotatingMarker

from preprocessing.aux import delay_preds, delay_preds_2d
from utils.nn import buildNN


class NeuralNet(BaseEstimator,ClassifierMixin):
    
    """ Ensembling with a Neural Network """
    
    def __init__(self,ensemble,architecture,training_params,
                 partsTrain=1,partsTest=1,
                 delay=4000,skip=100,jump=None,subsample=1,
                 smallEpochs=2,majorEpochs=20,checkEveryEpochs=2,
                 verbose=True):
        """Init."""
        ### timecourse history parameters ###
        # how many past time samples to include along with the most recent sample
        self.delay = delay
        # subsample above samples
        self.skip = skip
        # here can be set a custom subsampling scheme, it overrides previous params
        self.jump = jump
        
        ### RAM saving ###
        # due to RAM limitations the model is interchangeably trained on 'partsTrain' portions of the data
        self.partsTrain = partsTrain
        # also due to RAM limitations testing data has to be split into 'partsTest' parts
        self.partsTest = partsTest
        
        ### training ###
        # amounts of epochs to perform on the current portion of the training data
        self.smallEpochs = smallEpochs
        # amounts of major epochs to perform, 
        # i.e. on each major epoch a new portion of training data is obtained
        self.majorEpochs = majorEpochs
        # print AUC computed on test set every major epochs
        self.checkEveryEpochs = checkEveryEpochs
        
        # whether to calculate and print results during training
        self.verbose = verbose
        
        # used in bagging to set different starting points when subsampling the data
        self.mdlNr = 0
        
        self.subsample = subsample
        
        self.architecture = architecture
        self.ensemble = ensemble
        self.training_params = training_params
    
    def fit(self,X,y,Xtest=None,ytest=None):
        """Fit."""
        input_dim = X.shape[1]
        # set different data preparation schemes basing on what kind of NN is it
        layers = [i.keys()[0] for i in self.architecture]
        self.isCNN = 'Conv' in layers
        self.isRecurrent = 'GRU' in layers or 'LSTM' in layers        
        if self.isCNN:
            self.addDelay = delay_preds
            self.training_params['num_strides'] = self.delay//self.skip
        elif self.isRecurrent:
            self.addDelay = delay_preds_2d
        else:
            input_dim *= self.delay/self.skip
            input_dim = int( input_dim )
            self.addDelay = delay_preds
        
        # create the model
        self.model = buildNN(self.architecture, self.training_params, input_dim)
            
        widgets = ['Training : ', Percentage(), ' ', Bar(marker=RotatingMarker()),
           ' ', ETA(), ' ']
        pbar = ProgressBar(widgets=widgets, maxval=self.majorEpochs)
        pbar.start()
            
        # train the model on a portion of training data; that portion is changed each majorEpoch
        for majorEpoch in range(self.majorEpochs):
            startingPoint = majorEpoch%self.partsTrain or self.mdlNr%self.partsTrain
            if self.jump is not None:
                trainData = self.addDelay(X, delay=self.delay, skip=self.skip,
                                          subsample=self.partsTrain,start=startingPoint, jump=self.jump)
            else:
                trainData = self.addDelay(X, delay=self.delay, skip=self.skip,
                                          subsample=self.partsTrain,start=startingPoint)
                                         
            if self.isCNN:
                trainData = trainData.reshape((trainData.shape[0],1,trainData.shape[1],1))
            targets = y[startingPoint::self.partsTrain]
            
            trainData = trainData[::self.subsample]
            targets = targets[::self.subsample]
            
            self.model.fit(trainData, targets, nb_epoch=self.smallEpochs, 
                           batch_size=512,verbose=0,show_accuracy=True)
            
            trainData=None
            
            pbar.update(majorEpoch)
            
            if self.verbose and majorEpoch%self.checkEveryEpochs == 0:
                print("Total epochs: %d" % (self.smallEpochs*(majorEpoch+1)))
                if Xtest is not None and ytest is not None:
                    pred = self._predict_proba_train(Xtest)
                    score = np.mean(roc_auc_score(ytest[0::self.partsTest],pred))
                    print("Test AUC : %.5f" % (score))
                    pred = None
        
        if self.verbose:
            print('Training finished after %d epochs'% (self.smallEpochs*(majorEpoch+1)))
        
    def predict_proba(self,X):
        """Get predictions."""
        pred = []
        for part in range(self.partsTest):
            start = part*len(X)//self.partsTest-self.delay*(part>0)
            stop = (part+1)*len(X)//self.partsTest
            testData = self.addDelay(X[slice(start,stop)], delay=self.delay, skip=self.skip, 
                                       jump=self.jump)[self.delay*(part>0):]
            if self.isCNN:
                testData = testData.reshape((testData.shape[0],1,testData.shape[1],1))
            pred.append(self.model.predict_proba(testData, batch_size=512,verbose=0))
            testData = None
        pred = np.concatenate(pred)
        return pred
        
    def _predict_proba_train(self,X):
        """ Only used internally during training - subsamples test data for speed """
        testData = self.addDelay(X, delay=self.delay, skip=self.skip,subsample=self.partsTest,start=0,jump=self.jump)
        if self.isCNN:
            testData = testData.reshape((testData.shape[0],1,testData.shape[1],1))
        pred = self.model.predict_proba(testData, batch_size=512,verbose=0)
        testData = None
        return pred
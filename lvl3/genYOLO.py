# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:01:25 2015.

@author: rc,alex
"""
import os
import sys
if __name__ == '__main__' and __package__ is None:
    filePath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(filePath)

from glob import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from preprocessing.aux import getEventNames

cols = getEventNames()
ids = np.load('../infos_test.npy')
subject_test = ids[:, 1]
series_test = ids[:, 2]
ids = ids[:, 0]

labels = np.load('../infos_val.npy')
subjects = labels[:, -2]
series = labels[:, -1]
labels = labels[:, :-2]

subs_val = []
subs_test = []

# get all submissions
subs = glob('models/*.yml')
# remove folder and extension
subs = [sub[7:-4] for sub in subs]

# load subs
print 'Computing an average of %d submissions:' % len(subs)
for sub in subs:
    print sub
    subs_val.append(np.load('val/val_%s.npy' % sub)[0])
    subs_test.append(pd.read_csv('submissions/%s.csv' % sub,
                                 index_col=0).values)

# average all models
ens_val = np.mean(subs_val, axis=0)
ens_test = np.mean(subs_test, axis=0)

# stats of the average
aucs_val = [np.mean(roc_auc_score(labels[series == s], ens_val[series == s]))
            for s in [7, 8]]
print 'AUC: %.5f (SD: %.5f), s7: %.5f ; s8: %.5f' % (np.mean(aucs_val),
                                                     np.std(aucs_val),
                                                     aucs_val[0], aucs_val[1])

# save
np.save('val/val_YOLO.npy', [ens_val])
sub = pd.DataFrame(data=ens_test, index=ids, columns=cols)
sub.to_csv('submissions/YOLO.csv', index_label='id', float_format='%.8f')

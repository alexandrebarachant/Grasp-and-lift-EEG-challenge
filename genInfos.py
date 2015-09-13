# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 21:35:28 2015.

@author: fornax
"""
import numpy as np
import pandas as pd
from glob import glob
from mne import concatenate_raws

from preprocessing.aux import creat_mne_raw_object

# #### define lists #####
subjects = range(1, 13)

lbls_tot = []
subjects_val_tot = []
series_val_tot = []

ids_tot = []
subjects_test_tot = []
series_test_tot = []

# #### generate predictions #####
for subject in subjects:
    print 'Loading data for subject %d...' % subject
    # ############### READ DATA ###############################################
    fnames = glob('data/train/subj%d_series*_data.csv' % (subject))
    fnames.sort()
    fnames_val = fnames[-2:]

    fnames_test = glob('data/test/subj%d_series*_data.csv' % (subject))
    fnames_test.sort()

    raw_val = concatenate_raws([creat_mne_raw_object(fname, read_events=True)
                                for fname in fnames_val])
    raw_test = concatenate_raws([creat_mne_raw_object(fname, read_events=False)
                                for fname in fnames_test])

    # extract labels for series 7&8
    labels = raw_val._data[32:]
    lbls_tot.append(labels.transpose())

    # aggregate infos for validation (series 7&8)
    raw_series7 = creat_mne_raw_object(fnames_val[0])
    raw_series8 = creat_mne_raw_object(fnames_val[1])
    series = np.array([7] * raw_series7.n_times +
                      [8] * raw_series8.n_times)
    series_val_tot.append(series)

    subjs = np.array([subject]*labels.shape[1])
    subjects_val_tot.append(subjs)

    # aggregate infos for test (series 9&10)
    ids = np.concatenate([np.array(pd.read_csv(fname)['id'])
                         for fname in fnames_test])
    ids_tot.append(ids)
    raw_series9 = creat_mne_raw_object(fnames_test[1], read_events=False)
    raw_series10 = creat_mne_raw_object(fnames_test[0], read_events=False)
    series = np.array([10] * raw_series10.n_times +
                      [9] * raw_series9.n_times)
    series_test_tot.append(series)

    subjs = np.array([subject]*raw_test.n_times)
    subjects_test_tot.append(subjs)


# save validation infos
subjects_val_tot = np.concatenate(subjects_val_tot)
series_val_tot = np.concatenate(series_val_tot)
lbls_tot = np.concatenate(lbls_tot)
toSave = np.c_[lbls_tot, subjects_val_tot, series_val_tot]
np.save('infos_val.npy', toSave)

# save test infos
subjects_test_tot = np.concatenate(subjects_test_tot)
series_test_tot = np.concatenate(series_test_tot)
ids_tot = np.concatenate(ids_tot)
toSave = np.c_[ids_tot, subjects_test_tot, series_test_tot]
np.save('infos_test.npy', toSave)

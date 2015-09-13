# Grasp-and-Lift EEG Detection

Code and documentation for the winning solution in the Grasp-and-Lift EEG Detection challenge : https://www.kaggle.com/c/grasp-and-lift-eeg-detection

**Authors**:
* [Alexandre Barachant](http://alexandre.barachant.org)
* [Rafal Cycon](https://www.linkedin.com/pub/rafal-cycon/65/411/832)

**Contents** :

- [Signal Processing & Classification Pipeline ](#signal-processing-classification-pipeline)
    - [Overview](#overview)
    - [Models Description](#models-description)
        - [Level1](#level1)
        - [Level2](#level2)
        - [Level3](#level3)    
    - [Submissions](#submission)
    - [Discussion](#discussion)

- [Code](#code)
    - [Code overview](#code-overview)
    - [Generating submissions](#generating-submissions)
- [Appendix](#appendix)
    - [List of level1 models](#list-of-level1-models)
    - [List of level2 models](#list-of-level2-models)
    - [Source File description](#source-file-description)

**Licence** : BSD 3-clause. see Licence.txt

# Signal Processing & Classification Pipeline

## Overview

The goal of this challenge was to detect 6 different events related to hand
movement during a task of grasping and lifting an object, using only EEG signal.
We were asked to provide probabilities for the 6 events and for every time
sample. The evaluation metric for this challenge was the Area under ROC curve
(AUC) averaged over the 6 event types.

From an EEG point of view, brain patterns related to hand movement are
characterized by spatio-frequential change in EEG signal. More specifically, we
expect to see a decrease of signal power in the mu (12hz) frequency band over
the contralateral motor cortex coupled with an increase of power in the
ipsilateral motor-cortex.
Since these changes occur during/after the execution of the movement, and
considering that some events are labelled at the beginning of the movement (eg.
Hadstart) while some others the end of the movement (eg. Replace), it was very
difficult to build a single model scoring uniformly over the 6 different events.
In other words, depending on the event you try to classify, it was a prediction
problem, or a detection problem.

The 6 events were representing different stages of a sequence of hand movements
(hand starts moving, starts lifting the object, etc.). One of the challenges was
to take into account the temporal structure of the sequence i.e. the sequential
relationship between events. In addition, some events were overlapping, and some
others were mutually exclusive. As a result, it was difficult to follow a
multiclass approach to this problem, or to use Finite State Machine to decode
the sequence.

Finally, true labels were extracted from EMG signal, and provided as a +/-150ms
frame centered around the occurrence of the event. This 300ms frame has no particular physiological meaning i.e. there is no fundamental difference between a time sample at +150ms and another one at +151ms, while they have different labels. Therefore, another difficulty was to increase the sharpness of the prediction to minimize occurrence of false positives around edges of the frames.

In the above context, we built a 3-level classification pipeline:

- Level1 models are subject-specific, i.e. trained independently on each subject. Most of them are also event-specific. Their main goal is to provide support and diversity for level2 models by embedding subject and events specificities using different types of features.
- Level2 models are global models (i.e. not subject-specific) that are trained on level1 predictions (metafeatures). Their main goal is to take into account the temporal structure and relationship between events. Also the fact that they are global significantly helps to calibrate predictions between subjects.
- Level3 models ensemble level2 predictions via an algorithm that optimizes level2 models' weights to maximize AUC. This step improves the sharpness of predictions while reducing overfitting.

### No Future data rule

We took special care to ensure that causality was respected.
Every time a temporal filter is applied, we used the `lfilter` function of scipy
which implements a direct form II causal filtering.
Each time a sliding window was used, we padded the signal with zeros on the left. Same idea was applied when a history of past prediction was used.
Finally, since we concatenate signal or predictions across series before processing them, some of the last samples from previous series can "leak" into the next series. This is **not** a violation of the rule, since the rule only applies within a particular series.

## Models Description

In this section we provide an overview of the 3-level pipeline that was used in the solution.

### Level1

Models described below were trained on raw data, in *validation* and *test* modes. In the former models are trained on series 1-6 and predictions are produced on series 7-8 - these predictions are the training data (metafeatures) for level2 models. The latter mode trains models on series 1-8 and predicts on test series 9-10.

#### Features

**Cov** - Covariance matrices are features of choice for detection of hand movement from EEG. They contain spatial information (through correlation between channels) and frequential information (through variance of the signal). Covariance matrices were estimated using a sliding window (usually 500 samples) after bandpass filtering of the signal. We had two types of covariance features:
- 1) AlexCov : events' labels are first relabelled into a sequence of 7 (consecutive) brain states. For each brain state, the geometric mean of corresponding covariance matrices is estimated (according to a log-euclidean metric) and the Riemannian distance to each centroid is calculated, producing a feature vector of size 7. This procedure can be viewed as a supervised manifold embedding with a Riemannian metric.
- 2) RafalCov : same idea as in AlexCov but applied on each event separately, producing a 12 element feature vector (class 1 and class 0 for each event).

**ERP** - This dataset contained visual evoked potential associated with the experimental paradigm. Features for asynchronous ERP detection were essentially based on work done in the previous [BCI challenge](https://www.kaggle.com/c/inria-bci-challenge). During training, signal is epoched for 1 second preceding the onset of each event. ERPs are averaged and reduced with Xdawn algorithm before being concatenated to epoched signal. Then, covariances matrices are estimated and processed similarly to the covariance features.

**FBL** - the signal was found to contain lots of predictive information in low frequencies, therefore we introduced a "Filter Bank" approach. It consisted of concatenating together results from applying several 5th order Butterworth lowpass filters (cutoff frequencies at 0.5, 1, 2, 3, 4, 5, 7, 9, 15, 30 Hz) on the signal.

**FBL_delay** - as FBL, but a single row/observation is also augmented with 5 past samples that together spanned an interval of 2 seconds (1000 samples in the past, only taking each 200th sample). These additional features allow models to capture temporal structure of events.

**FBLC** - Filter Bank and Covariance matrices features concatenated together into a single feature set.

#### Algorithms
**Logistic Regression** and **LDA** (with different normalizations applied to train and test data prior to learning) built on above features provided an event-specific view on the data.

There were also two level1 Neural Network approaches that were not event-specific (i.e. trained on all events simultaneously):

**Convolutional Neural Network** - this family of models is based on [Tim Hochberg's script with Bluefool's tweaks](https://www.kaggle.com/bitsofbits/grasp-and-lift-eeg-detection/naive-nnet) slightly extended with lowpass filtering and optional 2D convolution (done across all electrodes, so that each filter captured simultaneous dependencies between all electrodes). In essence, this is a small 1D/2D convolutional neural network (input -> dropout -> 1D/2D conv -> dense -> dropout -> dense -> dropout -> output) that is trained on a current sample and a subsampled portion of past samples. Each CNN was bagged 100 times to reduce a relatively high variance in scores between each single run that most likely arose due to utilized memory-efficient epoching strategy that trained the net on random portions of training data.

**Recurrent Neural Network** - a small RNN (input -> dropout -> GRU -> dense -> dropout -> output) trained on a lowpass filtered signal (a Filter Bank composed of lowpass filters with cutoff frequencies at 1, 5, 10, 30 Hz). The model was trained on a sparse timecourse history of 8s (taking each 100th sample up to 4000 samples in the past). Even though RNNs seem to be perfectly suitable to the task at hand (clearly defined temporal structure of events and their interdependencies) we had troubles getting them to produce very accurate predictions, and due to high computational costs we did not pursue this approach as much as we would like to.

### Level2

These models are trained on predictions of level1 models. They were trained in *validation* and *test* modes. *Validation* was done in a cross-validation fashion, with splits done per series (2 folds). Predictions from folds are then metafeatures for level3 models. In the *test* mode models were trained jointly on series 7 and 8 and predictions were produced for test series 9 and 10.

#### Algorithms

**XGBoost** - gradient boosting machines provided an unique view on the data, achieving very good scores and providing diversity for next level models. They are also the only level2 models that were trained for each event separately and that had subjects' IDs added as a feature, which helped to calibrate predictions between subjects (adding one-hot encoding of subject's IDs in NN-based models did not improve their performances). The accuracy of XGBoost predicting correctly a particular event was much better when it was trained on a signal timecourse of several seconds and on metafeatures of all events rather than only metafeatures of the corresponding event, due to them extracting predictive information contained in interdependencies between events and associated temporal structure. Moreover, heavily subsampling input data served as regularization and greatly reduced overfitting.

**Recurrent Neural Network** - due to a clearly defined temporal structure of events and high diversity of metafeatures level2 RNNs were able to achieve a very high AUC, while also being computationally cheap when trained with the ADAM optimizer (in most cases only 1 epoch was required for convergence). A large number of level2 models are small modifications of a simple RNN architecture (input -> dropout -> GRU -> dense -> dropout -> output) that was trained on a subsampled signal timecourse of 8 seconds.

**Neural Network** - small multilayer neural networks (only a single hidden layer) were trained on a subsampled history timecourse of 3s. They achieved worse results than RNNs or XGBoost, but served as to provide diversity to level3 models.

**Convolutional Neural Network** - also small level2 CNNs (single convolution layer without any pooling, followed by a single dense layer) were trained on a subsampled history timecourse of 3s. Filters span all predictions for a single time sample and strides are made between time samples. As in the case of multilayer neural networks, the main purpose of these CNN models was to provide diversity to level3 models.

Diversity in level2 models was additionally enhanced by running above algorithms with following modifications between them:
- different subsets of metafeatures
- different length of timecourse history
- logarithmically sampled history (recent time points are more densely sampled instead of sampling at even intervals)
- adding bagging (see below)

And also for NNs/CNNs/RNNs:
- Parametric ReLU instead of ReLU as activation function in the dense layer
- multiple layers
- training with different optimizers (stochastic gradient descent or ADAM)

#### Bagging

Several models were additionally bagged to increase their robustness. Two kinds of bagging were used:

- selecting a random subset of training subjects for each bag (models that contain "bags" in their names),
- selecting a random subset of metafeatures for each bag (models that contain "bags_model" in their names); this methodology produced particularly well performing models.

In all cases using 15 bags was found to give satisfactory performance, and increasing the number of bags after that point brought insignificant increases in AUC.

### Level3

Level2 predictions are ensembled via an algorithm that optimizes ensembling weights to maximize AUC. This step improves the sharpness of predictions, and using a very simple ensembling method helps reduce overfitting, which was a real threat at such high ensembling level. To further increase AUC and robustness we used three types of weighted means:

- arithmetic mean
- geometric mean
- "power mean", in the form of $\bar x_p  = S(\sum x_i^{w_i})$, where $w_i=[0..3]$ and $S$ is a logistic function that is used to force output into [0..1] range.

A level3 model is an average of the above three weighted means.

## Submissions

Submission name | CV AUC | SD        | Public LB | Private LB
:--------------:|--------|-----------|-----------|-----------
"Safe1"         | 0.97831| 0.000014  | 0.98108 	 | **0.98095**
"Safe2"         | 0.97846| 0.000011  | 0.98117   | **0.98111**
"YOLO"          | 0.97881| 0.000143  | 0.98128   | **0.98109**

### "Safe1"

One of our final submissions is a level3 model that achieved a very stable and relatively high AUC in cross-validation. It was trained on a robust set of level2 metafeatures - 7 out of 8 level2 models were bagged.

### "Safe2"

An another submission with a very stable CV AUC. Its set of level2 metafeatures (only 6 level2 models out of 16 were bagged) caused us to (incorrectly) consider it to be a less safe pick than submission "Safe1" and was therefore **not selected** as a final submission. We include it in this write-up as a curiosity.

### "YOLO"

Our second and winning final submission was an average of 18 level3 models. Averaging them together showed an increased robustness in cross-validation scores as well as improvement in both CV and public LB scores. Note that this submission seems to be slightly overfitting, and it is possible to avoid its high computational costs by running either Safe1 or Safe2, both of which produce very similar AUC on the private leaderboard.

## Discussion

#### Are we decoding brain activity ?

Our models are based on a wide range of different features.
Whether these model are actually decoding brain activity related to hand movement is an open question. The use of complex preprocessing (for covariance feature) or black-box algorithm (CNN) bring additional difficulties when it comes to analyze results.

Good performances of models based on low frequency features bring some additional doubts. Indeed, these features are not known to be particularly useful for decoding of hand movement. More specifically, it is quite puzzling to achieve very good detection of a 300ms-wide event with frequency < 1Hz. One explanation could be change in baseline due to torso movement, or grounding when the subject touches the object.

We can also observe relatively good performances with a covariance model estimated on the 70-150Hz frequency range. This frequency range is too high to contain EEG, underlying the possible presence of task-related EMG activity.

However, the dataset is really clean, and contains very clear and strong pattern related to hand movement, as it can be seen in this [script](https://www.kaggle.com/alexandrebarachant/grasp-and-lift-eeg-detection/common-spatial-pattern-with-mne). While other activity (VEP, EMG etc.) may contribute to the overall performance by re-enforcing predictions for harder case, there is no doubt that we are indeed decoding brain activity related to hand movement.

#### Are all these models necessary ?

We made heavy use of ensembling for this challenge. The way the problem was defined (prediction of every sample) was playing in favor of this kind of solution. In this context, adding more models was always increasing the performance by boosting the sharpness of predictions.

For a real-life application, it is not really necessary to classify every time sample, and use a time frame approach, e.g output prediction for every 250ms time frame. We believe it is possible to obtain equivalent decoding performances with a much more optimized solution, stopping ensembling at level2, and using only a few subsets of level1 models (1 for each kind of features).

# Code

## Code overview

The source code is written in python. For each level,
you will find a folder containing models' description in the form of YAML files and several scripts to generate validation and test predictions. Each time, the predictions are saved in subfolders `val` and `test` and are used by the next level.

### Hardware
Throughout the challenge we used a workstation equipped with 12x3.5GHz cores, 64 GB RAM, 120GB SSD drive and Nvidia GeForce Titan Black. The code can be run on lower specs, but number of cores and RAM usage should be controlled via appropriate parameter values in models' YAML files (parameters `cores`, `partsTrain`, `partsTest` where applicable).

### Dependencies

- Python 2.7
- Numpy 1.9.2
- Scipy 0.16.0
- scikit-learn 0.17.dev0
- [pyriemann](https://github.com/alexandrebarachant/pyRiemann) 0.2.2 (from sources)
- [mne](https://github.com/mne-tools/mne-python) 0.10.dev0 (from sources)
- [XGBoost](https://github.com/dmlc/xgboost) 0.40 (from sources)
- Theano 0.7.0
- CUDA 7.0.27
- [Keras](https://github.com/fchollet/keras) 0.1.2 (from sources)
- [Lasagne](https://github.com/Lasagne/Lasagne) 0.2.dev1 (from sources)
- [nolearn](https://github.com/dnouri/nolearn) 0.6adev (from sources)
- [hyperopt](https://github.com/hyperopt/hyperopt) 0.0.2 (from sources)

Entirety of the code was ran on Ubuntu 14.04.

## Generating submissions

After installing the [dependencies](#dependencies), make sure you meet the [hardware](#hardware) requirement.

Data must be placed in the `data` folder, with training data in `data/train/` and test data in `data/test/`

The first thing to run is the genInfos script :
```
python genInfos.py
```

To generate level1 prediction, go to the lvl1 folder and run `genAll.sh`:
```
cd lvl1/
./genAll.sh
```
This script will take 3-4 days to run.


Next step is to run level2 predictions. Go to the lvl2 folder :

```
cd ../lvl2/
```
and you will have two options. You can either run all models (could take some time), or choose to run only models required for the [Safe1](#safe1) submission (recommended).

For all models run :

```
./genAll.sh
```

or, for the Safe1 submission, run :

```
./genSafe1.sh
```

This step takes 1-2 days to run.

We are almost there, go to the lvl3 folder:
```
cd ../lvl3/
```

and run one of the two available scripts.
For the [YOLO](#yolo) submission (requires generating all level2 models), run :
```
./genYOLO.sh
```
For the [Safe1](#safe1) submission, run :
```
./genSafe1.sh
```

generated submission is saved in the `submissions` subfolder.

# Appendix: performances of individual models

Here we provide a list of scores achieved individually by each model. For a few of them also leaderboard scores are given.
You can find description of preprocessing steps and settings for a model in its YAML file that is contained in *lvl[1-3]/models* folder.

## List of level1 models

**Nomenclature** : model names follow the following nomenclature: *[Feature]_[model]*. For covariance model, we have *C[window]_[frequency]_[model]*.

| Model name               | CV AUC | Public LB | Private LB |
|--------------------------|-------:|----------:|-----------:|
| FBL_L1                   | 0.7646 | - | -
| FBL_L2                   | 0.8647 | - | -
| FBL_Sc                   | 0.9001 | 0.92429 | 0.92297 	
| FBL_LDA                  | 0.9035 | - | -
| FBL_LDA_L1               | 0.9036 | - | -
|                          |        |
| FBL_delay_L1             | 0.7378 | - | -
| FBL_delay_L2             | 0.8844 | - | -
| FBL_delay_Sc             | 0.8978 | 0.93253 | 0.93302
| FBL_delay_LDA            | 0.9021 | - | -
|                          |        |
| FBLCA_L1                 | 0.7941 | - | -
| FBLCA_L2                 | 0.7971 | - | -
| FBLCA_Sc                 | 0.9155 | - | -
| FBLCA_LDA                | 0.9188 | 0.92700 | 0.93165
| FBLCA_LDA_L1             | 0.8897 | - | -
|                          |        |
| FBLCR_L1                 | 0.7815 | - | -
| FBLCR_L2                 | 0.7812 | - | -
| FBLCR_Sc                 | 0.9120 | - | -
| FBLCR_LDA                | 0.9136 | - | -
| FBLCR_LDA_L1             | 0.8933 | - | -
|                          |        |
| C500_[1_15]_LDA          | 0.8650 | - | -
| C500_[1_15]_LR           | 0.8733 | - | -
| C500_[1_15]_poly_LR      | 0.8823 | - | -
|                          |        |
| C500_[7_30]_LDA          | 0.8677 | - | -
| C500_[7_30]_LR           | 0.8686 | - | -
| C500_[7_30]_poly_LR      | 0.8791 | - | -
|                          |        |
| C500_[20_35]_LDA         | 0.8326 | - | -
| C500_[20_35]_LR          | 0.8402 | - | -
| C500_[20_35]_poly_LR     | 0.8481 | - | -
|                          |        |
| C500_[70_150]_LDA        | 0.8413 | - | -
| C500_[70_150]_LR         | 0.8508 | - | -
| C500_[70_150]_poly_LR    | 0.8401 | - | -
|                          |        |
| C250_[35]_LDA            | 0.8891 | - | -
| C250_[35]_LR             | 0.8975 | - | -
| C250_[35]_poly_LR        | 0.9065 | - | -
|                          |        |
| C500_[35]_LDA            | 0.8951 | - | -
| C500_[35]_LR             | 0.9069 | - | -
| C500_[35]_poly_LR        | 0.9151 | 0.92418 | 0.92416
|                          |        |
| ERPDist_LDA              | 0.8429 | - | -
| ERPDist                  | 0.8613 | - | -
| ERPDist_poly             | 0.8785 | - | -
|                          |        |
| CovsRafal_35Hz_256       | 0.8924 | - | -
| CovsRafal_35Hz_500       | 0.8965 | - | -
|                          |        |
| CAll_LR                  | 0.9322 | 0.94165 | 0.94141
| CAll_old_LR              | 0.9304 | - | -
|                          |        |
| RNN_FB_delay4000         | 0.8577 | 0.91375 | 0.91427 	
|                          |        |
| CNN_1D_FB30              | 0.9075 | 0.93845 | 0.94120 	
| CNN_1D_FB7-30            | 0.7579 | - | -
| CNN_1D_FB5               | 0.8995 | - | -
| CNN_1D_FB30_shorterDelay | 0.9001 |  - | -
|                          |        |
| CNN_2D_FB30              | 0.8986 | 0.95002 | 0.94910
| CNN_2D_FB30_shorterDelay | 0.8896 |  - | -

## List of level2 models

| Model name                                    | CV AUC |Public LB|Private LB |
|-----------------------------------------------|-------:|--------:|----------:|
| xgb_subjects_sub                              | 0.9736 | 0.97641 | 0.97561
| **xgb_bags**                                      | 0.9730 | 0.97546 | 0.97585
| **xgb_bags_model**                                | 0.9757 | 0.97806 | 0.97740
| **xgb_bags_delay**                                | 0.9747 | 0.97777 | 0.97729
| **xgb_short**                                     | 0.9725 | - | - |
| xgb_onlyCovs                                  | 0.9598 | - | - |
| xgb_noCovs                                    | 0.9707 | - | - |
| xgb_longshort                                 | 0.9747 | - | - |
| xgb_longshort_bags_model                      | 0.9762 | - | - |
| RNN_256PR_delay4000_allModels_ADAM            | 0.9752 | 0.97946 | 0.97946
| **RNN_256PR_delay4000_allModels_ADAM_bags_model** | 0.9768 | 0.97998 | 0.98013
| RNN_256_customDelay_allModels_ADAM            | 0.9722 | - | - |
| RNN_256_customDelay_allModels_ADAM_bags_model | 0.9750 | - | - |
| RNN_256_delay4000_allModels_ADAM              | 0.9756 | - | - |
| **RNN_256_delay4000_allModels_ADAM_bags**         | 0.9761 | - | - |
| **RNN_256_delay4000_allModels_ADAM_bags_model**   | 0.9763 | - | - |
| RNN_256_delay4000_allModels_ADAM_2layers      | 0.9741 | - | - |
| RNN_256_delay4000_allModels_ADAM_2layers_bags | 0.9752 | - | - |
| RNN_256_delay4000_allModelsWPoly_ADAM         | 0.9714 | - | - |
| RNN_256_delay4000_FBLCA                       | 0.9739 | - | - |
| RNN_256_delay4000_FBLCRAll_ADAM               | 0.9736 | - | - |
| RNN_256_delay4000_FBLCR_256_ADAM_lr           | 0.9735 | - | - |
| RNN_256_delay4000_FBLCR_256_oldCovsAll        | 0.9743 | - | - |
| RNN_256_delay4000                             | 0.9506 | - | - |
| **RNN_256_delay2000_allModels_ADAM_bags_model**   | 0.9762 | - | - |
| nn_256                                        | 0.9736 | - | - |
| nn_256_wpoly                                  | 0.9734 | - | - |
| NN_256_allModels_ADAM_bags                    | 0.9731 |  0.97680 | 0.97682
| NN_350_allModels_ADAM                         | 0.9735 | - | - |
| cnn_196                                       | 0.9700 | - | - |
| cnn_256                                       | 0.9716 | - | - |
| cnn_256_bags_model                            | 0.9749 | 0.97813 | 0.97796

Model in bold are used in the Safe1 submission

## Source File description

Complete description of classes and methods can be found in their corresponding file. Most of the classes inherit from sklearn's `ClassifierMixin` or `TransformerMixin` in order to be pipelined through sklearn's `Pipeline` framework.
Here is a short overview of the source code for each folder.

- **preprocessing**
  - *aux.py* : contains auxiliary methods and classes, mainly to read raw files, extract sliding windows, or augment input data with a portion of timecourse history.
  - *filterBank.py* : code for frequential filtering of the signal.
  - *covs_alex.py* : preprocessing for covariance-based features.
  - *erp.py* : preprocessing for ERP features.

- **utils**
  - *ensemble.py* : utility methods to read predictions generated in different levels.
  - *nn.py* : creating a neural network architecture as specified in the YAML file.

- **ensembling**
  - *NeuralNet.py* : sklearn compatible classifier for neural network models.
  - *WeightedMean.py* : sklearn compatible classifier for weighted mean ensembling with hyperopt parameter optimization.
  - *XGB.py* : sklearn compatible classifier for xgboost.

- **lvl1**
  - *genPreds.py* : script for FilterBank and covariance lvl1 models.
  - *genPreds_CNN_Tim.py* : script for CNN lvl1 models. This script is an adaptation of Tim Hochberg's CNN script.
  - *genPreds_RNN.py* : script for RNN lvl1 model.

- **lvl2**
  - *genEns.py* : script to generate ensembling of lvl1 models.
  - *genEns_BagsSubjects.py* : script for ensembling with bags of different subjects.
  - *genEns_BagsModels.py* : script for ensembling with bags of different lvl1 models.

- **lvl3**
  - *genFinal.py* : script for weighted mean ensembling of lvl2 models.
  - *genYOLO.py* : script to generate the YOLO submission.
"""Utils function for ensembling."""
import numpy as np


def loadPredictions(predictions, fileName, names, test=False, lvl=1):
    """Load prediction from file."""
    if test:
        prefix = '../lvl%d/test/test_' % lvl
    else:
        prefix = '../lvl%d/val/val_' % lvl
    temp = np.load(prefix+fileName+'.npy')
    assert(len(temp) == len(names))
    for i in range(len(temp)):
        predictions[names[i]] = temp[i]


def createEnsFunc(ensemble):
    """Create ensemble."""
    func = ''
    for model in ensemble:
        func += 'preds[\'%s\'],' % (model)
    ens = eval('lambda preds : np.c_[%s]' % (func))
    return ens


def getLvl1ModelList():
    """Get the complete list of lvl1 models."""
    # format: [filename, [model names]]
    files = [
            ['FBL', ['FBL_L1', 'FBL_L2', 'FBL_Sc', 'FBL_LDA', 'FBL_LDA_L1']],

            ['FBL_delay100_skip20',
            ['FBL_delay_L1', 'FBL_delay_L2',  'FBL_delay_Sc', 'FBL_delay_LDA']],

            ['FBLC_256pts_alex2',
            ['FBLCA_L1', 'FBLCA_L2', 'FBLCA_Sc', 'FBLCA_LDA', 'FBLCA_LDA_L1']],

            ['FBLCR_256',
            ['FBLCR_L1', 'FBLCR_L2', 'FBLCR_Sc', 'FBLCR_LDA', 'FBLCR_LDA_L1']],

            ['FBLCR_All', ['FBLCAll_Sc', 'FBLCAll_LDA_Sc']],

            ['CovsAlex_1-15Hz_500pts',   ['C500_[1_15]_LDA', 'C500_[1_15]_LR']],
            ['CovsAlex_7-30Hz_500pts',   ['C500_[7_30]_LDA', 'C500_[7_30]_LR']],
            ['CovsAlex_20-35Hz_500pts',  ['C500_[20_35]_LDA', 'C500_[20_35]_LR']],
            ['CovsAlex_70-150Hz_500pts', ['C500_[70_150]_LDA', 'C500_[70_150]_LR']],
            ['CovsAlex_35Hz_250pts', ['C250_[35]_LDA', 'C250_[35]_LR']],
            ['CovsAlex_35Hz_500pts', ['C500_[35]_LDA', 'C500_[35]_LR']],
            ['CovsERP_Dist', ['ERPDist_LDA', 'ERPDist']],

            ['CovsAlex_1-15Hz_500pts_poly',   ['C500_[1_15]_poly_LR']],
            ['CovsAlex_7-30Hz_500pts_poly',   ['C500_[7_30]_poly_LR']],
            ['CovsAlex_20-35Hz_500pts_poly',  ['C500_[20_35]_poly_LR']],
            ['CovsAlex_70-150Hz_500pts_poly', ['C500_[70_150]_poly_LR']],
            ['CovsAlex_35Hz_250pts_poly', ['C250_[35]_poly_LR']],
            ['CovsAlex_35Hz_500pts_poly', ['C500_[35]_poly_LR']],
            ['CovsERP_Dist_poly', ['ERPDist_poly']],

            ['CovAlex_All', ['CAll_LR']],
            ['CovAlex_old_All', ['CAll_old_LR']],
            ['CovsRafal_35Hz_256pts', ['CovsRafal_35Hz_256']],
            ['CovsRafal_35Hz_500pts', ['CovsRafal_35Hz_500']],

            ['RNN_FB_delay4000', ['RNN_FB_delay4000']],

            ['cnn_script_1D_30Hz', ['CNN_1D_FB30']],
            ['cnn_script_1D_7-30Hz', ['CNN_1D_FB7-30']],
            ['cnn_script_1D_5Hz', ['CNN_1D_FB5']],
            ['cnn_script_2D_30Hz', ['CNN_2D_FB30']],

            ['cnn_script_1D_30Hz_shorterDelay', ['CNN_1D_FB30_shorterDelay']],
            ['cnn_script_2D_30Hz_shorterDelay', ['CNN_2D_FB30_shorterDelay']],
            ]
    return files

# @INPROCEEDINGS{7746325,
#   author={Jayarathne, Isuru and Cohen, Michael and Amarakeerthi, Senaka},
#   booktitle={2016 IEEE 7th Annual Information Technology, Electronics and Mobile Communication Conference (IEMCON)}, 
#   title={BrainID: Development of an EEG-based biometric authentication system}, 
#   year={2016},
#   volume={},
#   number={},
#   pages={1-6},
#   doi={10.1109/IEMCON.2016.7746325}}

from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.metrics import det_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt

class CSP_LDA:
    def __init__(self):
        self.clf = LinearDiscriminantAnalysis()
        self.csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    def train(self, data, label):
        data = np.squeeze(data)
        # plt.plot(data[0,0,:])
        # plt.savefig("test.png")
        data = data.astype(np.float64)
        self.clf.fit(self.csp.fit_transform(data, label), label)

    def predict(self, data):
        data = np.squeeze(data)
        data = data.astype(np.float64)
        return self.clf.predict(self.csp.transform(data))

    def valid(self, data, label):
        data = np.squeeze(data)
        scores = self.clf.predict_proba(self.csp.transform(data))
        fpr, fnr, thresholds = det_curve(label, scores[:,1])
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print("EER: {}%".format(EER*100))
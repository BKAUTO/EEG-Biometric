# @article{kumar2017a,
#   author = {Kumar, Pradeep and Saini, Rajkumar and Pratim Roy, Partha and Prosad Dogra, Debi},
#   title = {A bio-signal based framework to secure mobile devices},
#   language = {eng},
#   format = {article},
#   journal = {Journal of Network and Computer Applications},
#   volume = {89},
#   pages = {62-71},
#   year = {2017},
#   issn = {10848045, 10958592},
#   publisher = {Academic Press},
#   doi = {10.1016/j.jnca.2017.02.011}
# }
from hmmlearn import hmm
import numpy as np
import operator
from copy import copy
from scipy.special import softmax

class HMM_classifier():
    def __init__(self, base_hmm_model):
        self.models = {}
        self.hmm_model = base_hmm_model

    def fit(self, X, Y):
        """
        X: input sequence [[[x1,x2,.., xn]...]]
        Y: output classes [1, 2, 1, ...]
        """
        print("Detect classes:", set(Y))
        print("Prepare datasets...")
        X_Y = {}
        X_lens = {}
        for c in set(Y):
            X_Y[c] = []
            X_lens[c] = []

        for x, y in zip(X, Y):
            X_Y[y].extend(x)
            X_lens[y].append(len(x))

        for c in set(Y):
            print("Fit HMM for", c, " class")
            hmm_model = copy(self.hmm_model)
            hmm_model.fit(X_Y[c], X_lens[c])
            self.models[c] = hmm_model

    def _predict_scores(self, X):

        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with log likehood per class
        """
        X_seq = []
        X_lens = []
        for x in X:
            X_seq.extend(x)
            X_lens.append(len(x))

        scores = {}
        for k, v in self.models.items():
            scores[k] = v.score(X)

        return scores

    def predict_proba(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: dict with probabilities per class
        """
        pred = self._predict_scores(X)

        keys = list(pred.keys())
        scores = softmax(list(pred.values()))

        return dict(zip(keys, scores))

    def predict(self, X):
        """
        X: input sample [[x1,x2,.., xn]]
        Y: predicted class label
        """
        pred = self.predict_proba(X)

        return max(pred.items(), key=operator.itemgetter(1))[0]

class HMM:
    def __init__(self):
        self.clf = HMM_classifier(hmm.GaussianHMM())
    
    def train(self, data, label):
        data = np.squeeze(data)
        self.clf.fit(data, label)

    def predict(self, data):
        data = np.squeeze(data)
        labels = []
        for i in range(data.shape[0]):
            labels.append(self.clf.predict(data[i,:,:]))
        return labels
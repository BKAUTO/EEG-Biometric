# @article{kumari2016a,
#   author = {Kumari Sharma, Pinki and Vaish, Abhishek},
#   title = {Individual identification based on neuro-signal using motor movement and imaginary cognitive process},
#   language = {eng},
#   format = {article},
#   journal = {Optik},
#   volume = {127},
#   number = {4},
#   pages = {2143-2148},
#   year = {2016},
#   issn = {16181336, 00304026},
#   publisher = {Elsevier GmbH},
#   doi = {10.1016/j.ijleo.2015.09.020}
# }
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import det_curve
import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.clf = MLPClassifier(hidden_layer_sizes=(64*24,32,2), solver='adam', random_state=1)

    def train(self, data, label):
        data = np.squeeze(data)
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
        self.clf.fit(data, label)
    
    def predict(self, data):
        data = np.squeeze(data)
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
        print(data.shape)
        return self.clf.predict(data)
    
    def valid(self, data, label):
        data = np.squeeze(data)
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
        scores = self.clf.predict_proba(data)
        fpr, fnr, thresholds = det_curve(label, scores[:,1])
        EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        print("EER: {}%".format(EER*100))
    
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
from sklearn import svm

# channel * 640 * 9

class SVM:
    def __init__(self):
        self.clf = svm.SVC()

    def train(self, data, label):
        self.clf.fit(data, label)

    def predict(self, data):
        return self.clf.predict(data)

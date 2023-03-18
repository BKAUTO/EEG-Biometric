# @article{autthasan2021a,
#   author = {Autthasan, Phairot and Chaisaen, Rattanaphon and Sudhawiyangkul, Thapanun and Kiatthaveephong, Suktipol and Rangpong, Phurin and Dilokthanakul, Nat and Bhakdisongkhram, Gun and Phan, Huy and Guan, Cuntai and Wilaiprasitporn, Theerawit},
#   title = {MIN2Net: End-to-End Multi-Task Learning for Subject-Independent Motor Imagery EEG Classification},
#   language = {eng},
#   format = {article},
#   journal = {Ieee Transactions on Biomedical Engineering},
#   volume = {PP},
#   number = {99},
#   pages = {1-1},
#   year = {2021},
#   issn = {15582531, 00189294},
#   publisher = {IEEE Computer Society},
#   doi = {10.1109/TBME.2021.3137184}
# }

from min2net.model import MIN2Net
import numpy as np
import matplotlib.pyplot as plt

class MIN2NET:
    def __init__(self):
        self.clf = MIN2Net(input_shape=(1, 624, 64), num_class=2, monitor='loss', shuffle=True, batch_size=8, lr=1e-4, min_lr=1e-6)
    
    def train(self, data, label):
        data = data[:,:,:624]
        plt.plot(data[0,0,:])
        plt.savefig("test1.png")
        data = np.swapaxes(data,1,2)
        data = np.expand_dims(data, axis=1)
        data_val = data[0:10]
        label_val = label[0:10]
        self.clf.fit(data, label, data_val, label_val)
    
    def predict(self, data, label):
        data = data[:,:,:624]
        data = np.swapaxes(data,1,2)
        data = np.expand_dims(data, axis=1)
        return self.clf.predict(data, label)[0]['y_pred']
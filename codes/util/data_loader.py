from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import pyedflib
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from picard import picard
from scipy.stats import skew, iqr
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA
from transforms import filterBank
# from transforms import MTF

from PDC import ChannelSelection

import tsia.plot


class BCI2aDataset(Dataset):
    def get_subject_data(self, subject, path, No_channels, No_trials, Window_Length, sample_ratio=1, label=1, train='train'):
        No_valid_trial = 0
        data_return = np.zeros((No_trials, No_channels, Window_Length))
        class_return = np.zeros(No_trials)

        if train == 'train':
            data = sio.loadmat(path+'A0'+str(subject)+'T.mat')['data']
        else:
            data = sio.loadmat(path+'A0'+str(subject)+'E.mat')['data']

        if subject == 4:
            start_index = 1
        else:
            start_index = 3
        for i in range(start_index, start_index+No_trials//48):
            run = [data[0,i][0,0]][0]
            run_X = run[0]
            run_trial = run[1]
            run_y = run[2]
            run_artif = run[5]

            for trial in range(0, run_trial.size//sample_ratio):
                if run_artif[trial] == 0:
                    data_return[No_valid_trial, :, :] = np.transpose(run_X[(int(run_trial[trial]+1.5*250)):(int(run_trial[trial]+6*250)), :No_channels])
                    class_return[No_valid_trial] = label 
                    No_valid_trial += 1
        
        data_return = data_return[0:No_valid_trial, :, :]
        class_return = class_return[0:No_valid_trial]

        return data_return, class_return

    def __init__(self, subject, path, train='train', transform=None):
        self.transform = transform
        No_channels = 22
        No_trials = 1*48
        Window_Length = int(4.5*250)
        
        if train == 'train':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, No_trials, Window_Length, sample_ratio=1, label=1, train=train)
            self.plot(self.data_return, "raw.png")
            # self.data_return = self.KMAR(self.data_return)
            # self.plot(self.data_return, "KMAR.png")
            for i in [x for x in range(1,6) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, No_trials, Window_Length, sample_ratio=3, label=0, train=train)
                # negative_data = self.KMAR(negative_data)
                self.data_return = np.concatenate((self.data_return, negative_data), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class), axis=0)
        elif train == 'intra_test':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, No_trials, Window_Length, sample_ratio=1, label=1, train=train)
            for i in [x for x in range(1,6) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, No_trials, Window_Length, sample_ratio=3, label=0, train=train)
                # negative_data = self.KMAR(negative_data)
                self.data_return = np.concatenate((self.data_return, negative_data), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class), axis=0)
        elif train == 'inter_test':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, No_trials, Window_Length, sample_ratio=1, label=1, train=train)
            for i in [x for x in range(6,10) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, No_trials, Window_Length, sample_ratio=3, label=0, train=train)
                # negative_data = self.KMAR(negative_data)
                self.data_return = np.concatenate((self.data_return, negative_data), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class), axis=0)
    
    def KMAR(self, data):
        data = np.transpose(data, (1, 0, 2))
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
        # K, W, Y = picard(data)
        data = np.transpose(data, (1, 0))
        ica = FastICA(n_components=data.shape[1])
        ica.fit(data)
        Y = ica.fit_transform(data)
        self.plot(Y, "ICA_component.png")
        Y_var = np.var(Y, axis=0, keepdims=True)
        Y_skew = skew(Y, axis=0).reshape((-1, Y.shape[1]))
        Y_iqr = iqr(Y, axis=0, interpolation='midpoint', keepdims=True)
        features = np.concatenate((Y_var, Y_skew, Y_iqr), axis=0)
        features = np.transpose(features, (1,0))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        artif_label = kmeans.labels_[np.argmax(Y_var)]
        for i, label in enumerate(kmeans.labels_):
            if (label == artif_label):
                Y[:, i] = 0
        data = ica.inverse_transform(Y)
        data = np.transpose(data, (1,0))
        data = np.reshape(data, (data.shape[0], data.shape[1]//1125, 1125))
        data = np.transpose(data, (1, 0, 2))
        return data

    def __len__(self):
        return self.data_return.shape[0]

    def __getitem__(self, index):
        data = self.data_return[index,:,:1124]
        if self.transform:
            data = self.transform(data)
        label = self.class_return[index]

        return data, int(label)
    
    def plot(self, data, title):
        # data = np.transpose(data, (1,0))
        fig, axs = plt.subplots(22, 1, figsize=(20,40))
        for i in range(22):
            if title == "ICA_component.png":
                axs[i].plot(data[:, i])
            else:
                axs[i].plot(data[44, i, 400:500])
        fig.savefig(title)

class PhysioDataset(Dataset):
    def get_subject_data(self, subject, path, No_channels, Window_Length=640, label=1, train='train'):
        if train == 'train':
            run = ["04", "06", "08", "10"]
        else:
            run = ["12", "14"]
        
        data_run = []

        for run_num in run:
            if (subject < 10):
                file = pyedflib.EdfReader(path+"/S00"+str(subject)+"/"+"S00"+str(subject)+"R"+run_num+".edf")
            else:
                file = pyedflib.EdfReader(path+"/S0"+str(subject)+"/"+"S0"+str(subject)+"R"+run_num+".edf")
            annotation = file.readAnnotations()
            marker = []
            for i in annotation[1]:
                marker.append(i*160)
            y = []
            for counter, dataPoints in enumerate(marker):
                for i in range(int(dataPoints)):
                    code = annotation[2][counter]
                    if code == 'T0':
                        y.append(0)
                    elif code == 'T1':
                        y.append(1)
                    elif code == 'T2':
                        y.append(2)
                    else:
                        #TODO
                        print("catch error here")
            totalSignals = file.signals_in_file #totalSignals = 64
            signal_labels = file.getSignalLabels() #label names of electrode in 10-10 system
            trial = 0
            trial_segment = []
            if y[0] != 0:
                trial = 1
                trial_segment.append(0)
            for i in range(1, len(y)):
                if y[i] != y[i-1] and y[i] != 0:
                    trial = trial + 1
                    trial_segment.append(i)
            data = np.zeros((trial, totalSignals, 640))
            for i in range(trial):
                for j in np.arange(totalSignals):
                    data[i, j, :] = file.readSignal(j)[trial_segment[i]:trial_segment[i]+640]
            data_run.append(data)
        
        data_return = data_run[0]
        for i in range(1, len(data_run)):
            data_return = np.concatenate((data_return, data_run[i]), axis=0)
        class_return = np.zeros(data_return.shape[0])
        for i in range(class_return.shape[0]):
            class_return[i] = label

        return data_return, class_return
    
    def __init__(self, subject, path, train='train', transform=None):
        self.transform = transform
        No_channels = 64

        if train == 'train':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, label=1, train=train)
            # self.plot(self.data_return, "raw.png")
            self.data_return = self.KMAR(self.data_return)
            # self.plot(self.data_return, "KMAR.png")
            channelSelection = ChannelSelection()
            channel_selected = channelSelection.select(self.data_return[20,:,:])
            np.savetxt('test.npy', self.data_return[20,:,:])
            b = np.loadtxt('test.npy')
            print(self.data_return[20,:,:]==b)
            print(channel_selected)
            for i in [x for x in range(1,6) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, label=0, train=train)
                negative_data = self.KMAR(negative_data)
                self.data_return = np.concatenate((self.data_return, negative_data[:20, :, :]), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class[:20]), axis=0)
        elif train == 'intra_test':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, label=1, train=train)
            for i in [x for x in range(1,6) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, label=0, train=train)
                negative_data = self.KMAR(negative_data)
                self.data_return = np.concatenate((self.data_return, negative_data), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class), axis=0)
        elif train == 'inter_test':
            self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, label=1, train=train)
            for i in [x for x in range(6,10) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, label=0, train=train)
                negative_data = self.KMAR(negative_data)
                self.data_return = np.concatenate((self.data_return, negative_data), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class), axis=0)

    def __len__(self):
        return self.data_return.shape[0]
    
    def __getitem__(self, index):
        data = self.data_return[index,:,:640]
        if self.transform:
            data = self.transform(data)
        label = self.class_return[index]

        return data, int(label)
    
    def KMAR(self, data):
        data = np.transpose(data, (1, 0, 2))
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))
        # K, W, Y = picard(data)
        data = np.transpose(data, (1, 0))
        ica = FastICA(n_components=data.shape[1])
        ica.fit(data)
        Y = ica.fit_transform(data)
        # self.plot(Y, "ICA_component.png")
        Y_ptp = np.ptp(Y, axis=0, keepdims=True)
        Y_skew = skew(Y, axis=0).reshape((-1, Y.shape[1]))
        Y_iqr = iqr(Y, axis=0, interpolation='midpoint', keepdims=True)
        features = np.concatenate((Y_ptp, Y_skew, Y_iqr), axis=0)
        features = np.transpose(features, (1,0))
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        artif_label = np.bincount(kmeans.labels_).argmin()
        for i, label in enumerate(kmeans.labels_):
            if (label == artif_label):
                Y[:, i] = 0
        data = ica.inverse_transform(Y)
        data = np.transpose(data, (1,0))
        data = np.reshape(data, (data.shape[0], data.shape[1]//640, 640))
        data = np.transpose(data, (1, 0, 2))
        return data

    def plot(self, data, title):
        # data = np.transpose(data, (1,0))
        fig, axs = plt.subplots(64, 1, figsize=(20,40))
        for i in range(64):
            if title == "ICA_component.png":
                axs[i].plot(data[:, i])
            else:
                axs[i].plot(data[44, i, :])
        fig.savefig(title)
    
if __name__ == '__main__':
    filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 160)

    train_data = PhysioDataset(1, "../../data/physionet/physionet.org/files/eegmmidb/1.0.0", train='train', transform=filterTransform)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
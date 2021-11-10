from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# from transforms import filterBank
# from transforms import MTF

import tsia.plot


class BCI2aDataset(Dataset):
    def get_subject_data(self, subject, path, No_channels, No_trials, Window_Length, sample_ratio=1, label=1, train=True):
        No_valid_trial = 0
        data_return = np.zeros((No_trials, No_channels, Window_Length))
        class_return = np.zeros(No_trials)

        if train:
            data = sio.loadmat(path+'A0'+str(subject)+'T.mat')['data']
        else:
            data = sio.loadmat(path+'A0'+str(subject)+'E.mat')['data']
       
        for i in range(3, 3+No_trials//48):
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

    def __init__(self, subject, path, train=True, transform=None):
        self.transform = transform
        No_channels = 22
        No_trials = 48
        Window_Length = int(4.5*250)

        self.data_return, self.class_return = self.get_subject_data(subject, path, No_channels, No_trials, Window_Length, sample_ratio=1, label=1, train=train)
        if train:
            for i in [x for x in range(1,6) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, No_trials, Window_Length, sample_ratio=4, label=0, train=train)
                self.data_return = np.concatenate((self.data_return, negative_data), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class), axis=0)
        else:
            for i in [x for x in range(6,10) if x != subject]:
                negative_data, negative_class = self.get_subject_data(i, path, No_channels, No_trials, Window_Length, sample_ratio=4, label=0, train=train)
                self.data_return = np.concatenate((self.data_return, negative_data), axis=0)
                self.class_return = np.concatenate((self.class_return, negative_class), axis=0)

        
        
    def __len__(self):
        return self.data_return.shape[0]

    def __getitem__(self, index):
        data = self.data_return[index,:,:1124]
        if self.transform:
            data = self.transform(data)
        label = self.class_return[index]

        return data, int(label)

if __name__ == '__main__':
    filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 250)
    mtf = MTF(bins=8, image_size=128)
    transform=transforms.Compose([filterTransform, mtf])
    train_data = BCI2aDataset(3, "../../data/", train=True, transform=transform)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

    data, label = next(iter(train_dataloader))
    # data = data.numpy().reshape(data.shape[1], data.shape[2], data.shape[3])
    # data = np.swapaxes(data, 1, 2)
    # plt.plot(data[0,8,:])
    # plt.savefig("data")
    data = np.squeeze(data.numpy())

    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    _, mappable_image = tsia.plot.plot_markov_transition_field(mtf=data[19,5], ax=ax, reversed_cmap=True)
    plt.colorbar(mappable_image)
    plt.savefig("mappable_image")
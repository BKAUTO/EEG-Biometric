import torch
import glob
import numpy as np
from torch import nn
from sklearn.metrics import det_curve, roc_curve
from torch.utils.data import DataLoader
from util.transforms import filterBank
from util.data_loader import BCI2aDataset, PhysioDataset
from model.fbcnet import FBCNet
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_path = "../data/physionet/physionet.org/files/eegmmidb/1.0.0"
# subject = 4
batch_size = 4
# model = "../trained/"+"4_*_50.pth"
# channels_file = "../trained/4.txt"

def val(dataloader, model, subject):
    net = FBCNet(nChan=10).to(device)
    net.load_state_dict(torch.load(glob.glob(model)[0], map_location=device))
    net.eval()
    scores = []
    all_y= []
    size = len(dataloader.dataset)
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            scores.extend(10**pred.cpu().numpy()[:,1:])
            all_y.extend(y.cpu().numpy())
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%\n")

    # fpr, tpr, thresholds = roc_curve(all_y, scores, pos_label=1)
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)
    # print(eer)
    # print(thresh)
   
    for i, score in enumerate(scores):
        scores[i] = score[0]
    print(scores)
    print(all_y)
    fpr, fnr, thresholds = det_curve(all_y, scores, pos_label=1)
    print(fpr.shape)
    print(fnr.shape)
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print("EER: {}%".format(EER*100))
    # EER = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    # print("EER: {}%".format(EER*100))

    plt.plot(fpr*100, fnr*100, lw=1, label='subject '+str(subject))
    # plt.legend()

    return EER

if __name__ == '__main__':
    filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 160)

    EER = []

    plt.plot([0, 20], [0, 20], '--', color=(0.6, 0.6, 0.6))
    plt.title('Detection Error Tradeoff (DET) curves')
    plt.grid(linestyle='--')
    plt.xlim([0,20])
    plt.ylim([0,20])
    plt.xlabel("fpr(%)")
    plt.ylabel("fnr(%)")

    for subject in range(1,11):
        model = "../trained/"+"50/"+str(subject)+"_*_50.pth"
        channels_file = "../trained/"+"50/"+str(subject)+".txt"

        # define an empty list
        channels = []
        # open file and read the content in a list
        with open(channels_file, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                # add item to the list
                channels.append(int(currentPlace))
        
        intra_test_data = PhysioDataset(subject=subject, path=data_path, train="intra_test", transform=filterTransform, channels=channels)
        intra_test_dataloader = DataLoader(intra_test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        # inter_test_data = PhysioDataset(subject=subject, path=data_path, train="inter_test", transform=filterTransform, channels=channels)
        # inter_test_dataloader = DataLoader(inter_test_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        EER.append(val(intra_test_dataloader, model, subject))
    
    with open('EER.txt', 'w') as filehandle:
        for i in EER:
            filehandle.write('%s\n' % i)
    
    plt.legend()
    plt.savefig("test_result")
    print(EER)
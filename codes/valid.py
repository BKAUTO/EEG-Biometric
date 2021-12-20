import torch
import numpy as np
from torch import nn
from sklearn.metrics import det_curve, roc_curve
from torch.utils.data import DataLoader
from util.transforms import filterBank
from util.data_loader import BCI2aDataset
from model.fbcnet import FBCNet
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

data_path = "../data/"
subject = 3
batch_size = 4
outer = "outer"
epoch = 193
model = "../trained/"+"1_98.0_98.0.pth"

def val(dataloader, model):
    net = FBCNet(nChan=10).to(device)
    net.load_state_dict(torch.load(model, map_location=device))
    net.eval()
    scores = []
    all_y= []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = net(X)
            scores.extend(10**pred.cpu().numpy()[:,1:])
            all_y.extend(y.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_y, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    print(eer)
    print(thresh)
    
    fpr, fnr, thresholds = det_curve(all_y, scores)
    # EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # print("EER: {}%".format(EER*100))
    # EER = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
    # print("EER: {}%".format(EER*100))

    plt.plot(fpr*100, fnr*100, lw=1)
    plt.plot([0, 20], [0, 20], '--', color=(0.6, 0.6, 0.6))
    plt.title('Detection Error Tradeoff (DET) curves')
    plt.grid(linestyle='--')
    plt.xlim([0,20])
    plt.ylim([0,20])
    plt.xlabel("fpr(%)")
    plt.ylabel("fnr(%)")
    # plt.legend()
    plt.savefig("test_result")

if __name__ == '__main__':
    filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 160)
    test_data = BCI2aDataset(subject=subject, path=data_path, train=False, transform=filterTransform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val(test_dataloader, model)
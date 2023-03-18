from numpy import fabs
from numpy.core.numeric import outer
from util.data_loader import BCI2aDataset, PhysioDataset
from util.transforms import filterBank, gammaFilter, MSD, Energy_Wavelet, Normal
from torchvision import transforms
from torch.utils.data import DataLoader
from model.svm import SVM
from model.hmm import HMM
from model.energyNN import NeuralNetwork
from model.csp_lda import CSP_LDA
from model.mixed_fbcnet import MIXED_FBCNet
from model.cp_mixednet import CP_MixedNet
from model.mi_cnn import MI_CNN
from model.cnn_lstm import CNNLSTM
# from model.min2net import MIN2NET
from model.eegnet import EEGNet
from model.fbcnet import FBCNet
from torch import nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, det_curve

import os
import torch
import numpy as np
import sys
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# folder to load config file
CONFIG_PATH = "../config/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, dtype=torch.float), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%\n")

def train_svm(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    model.train(data, label)
    print("Train Completed.")

def test_svm(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    predicted = model.predict(data)
    print("Test Error: \n Accuracy: {}%".format(accuracy_score(label, predicted)*100))
    model.valid(data, label)

def train_hmm(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    model.train(data, label)
    print("Train Completed.")

def test_hmm(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    predicted = model.predict(data)
    print("Test Error: \n Accuracy: {}%".format(accuracy_score(label, predicted)*100))

def train_NN(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    model.train(data, label)
    print("Train Completed.")

def test_NN(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    predicted = model.predict(data)
    print("Test Error: \n Accuracy: {}%".format(accuracy_score(label, predicted)*100))
    model.valid(data, label)

def train_LDA(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    model.train(data, label)
    print("Train Completed.")

def test_LDA(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    predicted = model.predict(data)
    print("Test Error: \n Accuracy: {}%".format(accuracy_score(label, predicted)*100))
    model.valid(data, label)

def train_MIN2(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    model.train(data, label)
    print("Train Completed.")

def test_MIN2(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    predicted = model.predict(data, label)
    print(predicted)
    print("Test Error: \n Accuracy: {}%".format(accuracy_score(label, predicted)*100))

def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    epoch_acc = 100.0*correct
    return epoch_acc

def valid(dataloader, model):
    model.eval()
    scores = []
    all_y= []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, dtype=torch.float), y.to(device)
            pred = model(X)
            scores.extend(10**pred.cpu().numpy()[:,1:])
            all_y.extend(y.cpu().numpy())
    for i, score in enumerate(scores):
        scores[i] = score[0]
    fpr, fnr, thresholds = det_curve(all_y, scores, pos_label=1)
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    print("EER: {}%".format(EER*100))

if __name__ == '__main__':
    config = load_config(sys.argv[1])

    if config["transform"]["name"] == "nBand":    
        filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 160)
    
    elif config["transform"]["name"] == "filter_msd":
        filterTransform = transforms.Compose([gammaFilter(), MSD()])

    elif config["transform"]["name"] == "gammaFilter":
        filterTransform = gammaFilter()
    
    elif config["transform"]["name"] == "Energy_Wavelet":
        filterTransform = Energy_Wavelet()
    
    elif config["transform"]["name"] == "AlphaBeta":
        filterTransform = gammaFilter(band=[8,30])
    
    elif config["transform"]["name"] == "Alpha":
        filterTransform = gammaFilter(band=[8,12])
    
    elif config["transform"]["name"] == "Normal":
        filterTransform = Normal()
    
    elif config["transform"]["name"] == "None":
        filterTransform = None
    
    if config["dataset"]["name"] == "PhysioDataset":
        train_data = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="train", transform=filterTransform, select_channel=config["channel"]["select"], use_channel_no=config["channel"]["number"], preprocess=config["dataset"]["preprocess"])
        if config["train"]["batch_size"] == "all":
            batch_size = len(train_data)
        else:
            batch_size = config["train"]["batch_size"]
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) 
        
        channels = train_data.channels()

        intra_test_data = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="intra_test", transform=filterTransform, channels=channels, preprocess=config["dataset"]["preprocess"])
        if config["evaluate"]["batch_size"] == "all":
            batch_size = len(intra_test_data)
        else:
            batch_size = config["evaluate"]["batch_size"]
        intra_test_dataloader = DataLoader(intra_test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        inter_test_data = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="inter_test", transform=filterTransform, channels=channels, preprocess=config["dataset"]["preprocess"])
        if config["evaluate"]["batch_size"] == "all":
            batch_size = len(inter_test_data)
        else:
            batch_size = config["evaluate"]["batch_size"]
        inter_test_dataloader = DataLoader(inter_test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    elif config["dataset"]["name"] == "BCI2aDataset":
        train_data = BCI2aDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="train", transform=filterTransform, select_channel=config["channel"]["select"])
        if config["train"]["batch_size"] == "all":
            batch_size = len(train_data)
        else:
            batch_size = config["train"]["batch_size"]
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) 

        channels = train_data.channels()

        intra_test_data = BCI2aDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="intra_test", transform=filterTransform, channels=channels)
        if config["evaluate"]["batch_size"] == "all":
            batch_size = len(intra_test_data)
        else:
            batch_size = config["evaluate"]["batch_size"]
        intra_test_dataloader = DataLoader(intra_test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        inter_test_data = BCI2aDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="inter_test", transform=filterTransform, channels=channels)
        if config["train"]["batch_size"] == "all":
            batch_size = len(train_data)
        else:
            batch_size = config["train"]["batch_size"]
        inter_test_dataloader = DataLoader(inter_test_data, batch_size=batch_size, shuffle=True, drop_last=True)
 
    # save channels
    with open(str(config["train"]["subject"])+'.txt', 'w') as filehandle:
        for channel in channels:
            filehandle.write('%s\n' % channel)

    if config["model"]["name"] in ["FBCNet", "MICNN", "CNN_LSTM", "CP_MIXEDNET", "EEGNet", "MIXED_FBCNet"]:
        if config["model"]["name"] == "MIXED_FBCNet":
            model = MIXED_FBCNet(nChan=config["channel"]["number"]).to(device)
        
        elif config["model"]["name"] == "MICNN":
            model = MI_CNN(nChan=config["channel"]["number"]).to(device)
        
        elif config["model"]["name"] == "CNN_LSTM":
            model = CNNLSTM().to(device)
        
        elif config["model"]["name"] == "CP_MIXEDNET":
            model = CP_MixedNet().to(device)
        
        elif config["model"]["name"] == "EEGNet":
            model = EEGNet().to(device)
        
        elif config["model"]["name"] == "FBCNet":
            model = FBCNet(nChan=config["channel"]["number"]).to(device)

        loss_fn = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["initial_lr"], weight_decay=config["optimizer"]["weight_decay"])
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
        best_intra_acc = 0
        best_inter_acc = 0

        for t in range(config["train"]["epochs"]):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            exp_lr_scheduler.step()
            if (t > 50):
                intra_acc = test(intra_test_dataloader, model, loss_fn, t)
                inter_acc = test(inter_test_dataloader, model, loss_fn, t)
                if t > 250:
                    best_inter_acc = inter_acc
                    valid(intra_test_dataloader, model)
                    valid(inter_test_dataloader, model)
                #     torch.save(model.state_dict(), "../trained/"+str(config["train"]["subject"])+"_"+str(intra_acc)+"_"+str(inter_acc)+"_20"+".pth")
        print("Done!")

    elif config["model"]["name"] == "SVM":
        model = SVM()
        train_svm(train_dataloader, model)
        test_svm(intra_test_dataloader, model)
        test_svm(inter_test_dataloader, model)
    
    elif config["model"]["name"] == "HMM":
        model = HMM()
        train_hmm(train_dataloader, model)
        test_hmm(intra_test_dataloader, model)
        test_hmm(inter_test_dataloader, model)
    
    elif config["model"]["name"] == "NeuralNetwork":
        model = NeuralNetwork()
        train_NN(train_dataloader, model)
        test_NN(intra_test_dataloader, model)
        test_NN(inter_test_dataloader, model)
    
    elif config["model"]["name"] == "CSP_LDA":
        model = CSP_LDA()
        train_LDA(train_dataloader, model)
        test_LDA(intra_test_dataloader, model)
        test_LDA(inter_test_dataloader, model)
    
    elif config["model"]["name"] == "MIN2Net":
        model = MIN2NET()
        train_MIN2(train_dataloader, model)
        test_MIN2(intra_test_dataloader, model)
        test_MIN2(inter_test_dataloader, model)
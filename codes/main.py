from numpy import fabs
from numpy.core.numeric import outer
from util.data_loader import BCI2aDataset, PhysioDataset
from util.transforms import filterBank
from torch.utils.data import DataLoader
from model.svm import SVM
from model.mixed_fbcnet import FBCNet
from model.cp_mixednet import CP_MixedNet
import torch
from torch import nn
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score

import os
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

config = load_config("config.yaml")

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        print(X.shape)
        print(y.shape)
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

def test_svm(dataloader, model):
    data, label = next(iter(dataloader))
    data = data.numpy()
    label = label.numpy()
    predicted = model.predict(data)
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

if __name__ == '__main__':
    if config["transform"]["name"] == "nBand":    
        filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 160)
    
    if config["dataset"]["name"] == "PhysioDataset":
        train_data = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="train", transform=filterTransform, use_channel_no=config["channel"]["number"])
        train_dataloader = DataLoader(train_data, batch_size=config["train"]["batch_size"], shuffle=True, drop_last=True) 
        
        channels = train_data.channels()

        intra_test_data = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="intra_test", transform=filterTransform, channels=channels)
        intra_test_dataloader = DataLoader(intra_test_data, batch_size=config["evaluate"]["batch_size"], shuffle=True, drop_last=True)

        inter_test_data = PhysioDataset(subject=config["train"]["subject"], path=config["dataset"]["location"], train="inter_test", transform=filterTransform, channels=channels)
        inter_test_dataloader = DataLoader(inter_test_data, batch_size=config["evaluate"]["batch_size"], shuffle=True, drop_last=True)

    elif config["dataset"]["name"] == "BCI2aDataset":
        train_data = BCI2aDataset(subject=subject, path=data_path, train="train", transform=filterTransform)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True) 

        channels = train_data.channels()

        intra_test_data = BCI2aDataset(subject=subject, path=data_path, train="intra_test", transform=filterTransform, channels=channels)
        intra_test_dataloader = DataLoader(intra_test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        inter_test_data = BCI2aDataset(subject=subject, path=data_path, train="inter_test", transform=filterTransform, channels=channels)
        inter_test_dataloader = DataLoader(inter_test_data, batch_size=batch_size, shuffle=True, drop_last=True)
 
    # save channels
    with open(str(config["train"]["subject"])+'.txt', 'w') as filehandle:
        for channel in channels:
            filehandle.write('%s\n' % channel)

    if config["model"]["name"] == "FBCNet":
        model = FBCNet(nChan=config["channel"]["number"]).to(device)

        loss_fn = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["initial_lr"], weight_decay=config["optimizer"]["weight_decay"])
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        best_intra_acc = 0
        best_inter_acc = 0

        for t in range(config["train"]["epochs"]):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            exp_lr_scheduler.step()
            if (t > 50):
                intra_acc = test(intra_test_dataloader, model, loss_fn, t)
                inter_acc = test(inter_test_dataloader, model, loss_fn, t)
                if t > 250 and inter_acc >= best_inter_acc:
                    best_inter_acc = inter_acc
                    torch.save(model.state_dict(), "../trained/"+str(config["train"]["subject"])+"_"+str(intra_acc)+"_"+str(inter_acc)+"_20"+".pth")
        print("Done!")

    if config["model"]["name"] == "SVM":
        model = SVM()
        model.train
from numpy import fabs
from numpy.core.numeric import outer
from util.data_loader import BCI2aDataset
from util.transforms import filterBank
from torch.utils.data import DataLoader
from model.fbcnet import FBCNet
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

learning_rate = 0.0001
batch_size = 4
subject = 3
data_path = "../data/"
epochs = 200
trained_model_path = "../trained/"

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if epoch > epochs*0.95:
        torch.save(model.state_dict(), trained_model_path+"{}_trained_{}.pth".format(subject, epoch))

if __name__ == '__main__':
    filterTransform = filterBank([[4,8],[8,12],[12,16],[16,20],[20,24],[24,28],[28,32],[32,36],[36,40]], 250)
    
    train_data = BCI2aDataset(subject=subject, path=data_path, train=True, transform=filterTransform)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 

    test_data = BCI2aDataset(subject=subject, path=data_path, train=False, transform=filterTransform)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    model = FBCNet(nChan=22).to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer_final = torch.optim.Adam(model.parameters(), lr=learning_rate*0.1)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # if t < epochs*0.8:
        train(train_dataloader, model, loss_fn, optimizer)
        # else:
        #     train(train_dataloader, model, loss_fn, optimizer_final)
        test(test_dataloader, model, loss_fn, t)
    print("Done!")





# X = torch.rand(4, 1, 22, 1124, 9, device=device)
# model = FBCNet(nChan=22)
# logits = model(X)
# print(logits)
# target = torch.tensor([1,1,1,1])
# loss = nn.NLLLoss()
# output = loss(logits, target)
# y_pred = logits.argmax(1)
# print(f"Predicted class: {y_pred}")
# print(output)
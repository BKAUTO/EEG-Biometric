import torch
import numpy as np
from torch.utils.data import DataLoader

class Evaluate(object):
    def __init__(self, model, train_dataloader, device):
        embeddings_train = np.array([], dtype=np.int64).reshape(0,128)
        label_train = np.array([], dtype=np.int64).reshape(0)
        model.eval()
        with torch.no_grad():
            for X, y in train_dataloader:
                X, y = X.to(device), y.to(device)
                embedding = model(X).detach().cpu().numpy()
                embeddings_train = np.concatenate((embeddings_train, embedding), axis=0)
                label_train = np.concatenate((label_train, y.detach().cpu().numpy()), axis=0)
        self.embeddings_train = embeddings_train
        print(self.embeddings_train.shape)
        self.label_train = label_train
        print(self.label_train.shape)
        print("evaluate prepare done.")

    def calculate_pred_label(self, embeddings_test, label_test):
        correct = 0
        embeddings_test = embeddings_test.detach().cpu().numpy()
        label_test = label_test.detach().cpu().numpy()
        for i in range(embeddings_test.shape[0]):
            distance_pos = 0
            distance_neg = 0
            for j in range(self.embeddings_train.shape[0]):
                if (label_test[i].item() == self.label_train[j].item()):
                    distance_pos += np.linalg.norm(embeddings_test[i]-self.embeddings_train[j])
                else:
                    distance_neg += np.linalg.norm(embeddings_test[i]-self.embeddings_train[j])
            if (distance_pos >= distance_neg):
                correct += 1
        print(correct)
        return correct

                




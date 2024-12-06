import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import neighbors
import torch
from torch import nn
import torch.nn.functional as functional
import copy
from torch.utils.data import Dataset, DataLoader
import time

class CustomDataset(Dataset):
    def __init__(self, pandas_dataframe, y_cols):
        self.ys = pandas_dataframe[y_cols]
        self.xs = pandas_dataframe.drop(y_cols, axis=1)

    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        y = self.ys.iloc[idx]
        x = torch.Tensor(np.array(self.xs.iloc[idx]))
        return x, y

def preprocessing(df):
    new_df = df.drop(['Name'], axis=1)
    new_df = new_df.drop(['PassengerId'], axis=1)
    new_df['CryoSleep'] = pd.to_numeric(new_df['CryoSleep'])
    new_df['VIP'] = pd.to_numeric(new_df['VIP'])
    new_df['Destination'] = pd.factorize(new_df['Destination'])[0]
    new_df['HomePlanet'] = pd.factorize(new_df['HomePlanet'])[0]
    new_df['Cabin'] = pd.factorize(new_df['Cabin'])[0]
    for col in new_df.columns.values:
        new_df[col] = new_df[col].fillna((new_df[col].median()))
    try:
        new_df['Transported'] = new_df['Transported'].astype(float)
    except KeyError:
        pass
        # print("Skipping target column for test dataframe")
    return new_df

def see_correlation(df: pd.DataFrame, target: str, analyze_cols: list[str] = None):
    corrs = []
    analyze_cols = list(df.columns.values) if analyze_cols is None else analyze_cols
    for col in analyze_cols:
        corr = df[col].corr(df[target])
        corrs.append(corr)
    plt.bar(analyze_cols, corrs)
    plt.xticks(rotation=90)
    plt.show()

def knn_param_search(train_data, train_labels, test_data, test_labels,
                     metrics=('manhattan', 'euclidean', 'chebyshev'), 
                     ks=(1, 3, 5, 10, 25, 50, 100, 250), algorithm='brute'):
    """
    Takes a dataset and plots knn classification accuracy 
    for different hyper parameters.

    n_train and n_test allows to subsample the dataset for faster iteration
    """
    x_train = np.array(train_data)
    y_train = np.array(train_labels)
    x_test = np.array(test_data)
    y_test = np.array(test_labels)
    max_acc = 0
    datas = []
    for metric in metrics:
        for k in ks:
            print(f'Metric: {metric}; k: {k:3};', end=' ')
            classifier = neighbors.KNeighborsClassifier(k, algorithm=algorithm, metric=metric)
            classifier = classifier.fit(x_train, y_train)

            labels = classifier.predict(x_test)
            
            correct = labels == np.array(y_test)
            print(f'Accuracy: {correct.mean() * 100:.2f}%')
            if correct.mean() > max_acc:
                max_acc = correct.mean()
                best_classifier = classifier
                best_metric = metric
                best_k = k
            datas.append([metric, k, correct.mean()])
            
    print(f'Best classifier | metric: {best_metric}; k: {best_k:3}; accuracy: {max_acc * 100:.2f}%')
    return best_classifier, datas

class MLP(nn.Module):
    def __init__(self, n_hidden_neurons: int, lenght_x: int, output_layer_lenght: int):
        super().__init__()
        self.fc1 = nn.Linear(lenght_x, n_hidden_neurons)
        self.fc2 = nn.Linear(n_hidden_neurons, output_layer_lenght)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=1)
        h = functional.relu(self.fc1(x))
        logits = self.fc2(h)
        return logits
  
    def predict(self, Xs):
        predictions = []
        for x in Xs:
            x = torch.Tensor(x.reshape((1, x.shape[0])))
            logits = self(x)
            log_probs = functional.log_softmax(logits, dim=1)
            predicted_classes = torch.argmax(log_probs, dim=1)
            predictions.append(predicted_classes.bool().item())
        return predictions

def test_mlps(loader_train, loader_valid, epochs=50, lrs=[1e-1, 1e-2, 1e-3], hidden_neurons=[10, 100, 1000]):
    all_losses, all_accuracies = {}, {}
    best_accuracy = 0
    best_model = None

    for lr in lrs:
        for n_hidden_neurons in hidden_neurons:
            model = MLP(n_hidden_neurons, loader_train.dataset[0][0].shape[0], 2)
            opt = torch.optim.SGD(model.parameters(), lr=lr)
            train_losses, train_accuracies = [], []
            valid_losses, valid_accuracies = [], []
            start = time.time()
            for epoch in range(epochs):
                print(f"Learning rate: {lr}; N hidden neurons: {n_hidden_neurons}; Epoch: {epoch+1}/{epochs}", end='\r')
                # train
                epoch_losses = []
                correct, total = 0, 0
                for x, y in loader_train:
                    y = y.type(torch.LongTensor)
                    opt.zero_grad()
                    logits = model(x)  # logits: common name for the output before softmax activation
                    log_probs = functional.log_softmax(logits, dim=1)  # numerically stable version of log(softmax(logits))
                    loss = functional.nll_loss(log_probs, y)  # negative log likelihood loss
                    # or just: loss = F.cross_entropy(logits, y)
                    
                    loss.backward()
                    opt.step()
                    
                    epoch_losses.append(loss.item())
                    total += len(x)
                    correct += (torch.argmax(logits, dim=1) == y).sum().item()
                train_losses.append(np.mean(epoch_losses))
                train_accuracies.append(correct / total)
                
                # valid
                epoch_losses = []
                correct, total = 0, 0
                for x, y in loader_valid:
                    y = y.type(torch.LongTensor)
                    with torch.no_grad():
                        logits = model(x)
                    loss = functional.cross_entropy(logits, y)

                    epoch_losses.append(loss.item())
                    total += len(x)
                    correct += (torch.argmax(logits, dim=1) == y).sum().item()
                valid_losses.append(np.mean(epoch_losses))
                valid_accuracy = correct / total
                valid_accuracies.append(valid_accuracy)
                
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    best_model = copy.deepcopy(model), n_hidden_neurons, epoch
            print(f"Learning rate: {lr}; N hidden neurons: {n_hidden_neurons}; Epoch: {epoch+1}/{epochs}; Time: {time.time()-start:02.2f}[sec]; valid accuracy mean: {np.array(valid_accuracies).mean():.4f}; valid accuracy max: {np.array(valid_accuracies).max():.4f}")
            print("-"*100)        
            all_losses[str(lr) + " " + str(n_hidden_neurons)] = train_losses, valid_losses
            all_accuracies[str(lr) + " " + str(n_hidden_neurons)] = train_accuracies, valid_accuracies

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    for n, (train_losses, valid_losses) in all_losses.items():
        p = plt.plot(train_losses, label=f'{n}:train')
        plt.plot(valid_losses, label=f'{n}:valid', ls='--', c=p[0].get_color())
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    for n, (train_accuracies, valid_accuracies) in all_accuracies.items():
        p = plt.plot(train_accuracies, label=f'{n}:train')
        plt.plot(valid_accuracies, label=f'{n}:valid', ls='--', c=p[0].get_color())
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

    # for net in all_accuracies:
    #     train_accuracy = np.array(all_accuracies[net][0])
    #     valid_accuracy = np.array(all_accuracies[net][1])
    #     print(f"{net:10s} train accuracy max: {train_accuracy.max():.4f} train accuracy mean: {train_accuracy.mean():.4f} valid accuracy max: {valid_accuracy.max():.4f} valid accuracy mean: {valid_accuracy.mean():.4f}")
    return best_model


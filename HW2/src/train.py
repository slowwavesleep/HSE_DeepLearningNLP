import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score



def train_model(model: nn.Module,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                criterion,
                scheduler,
                gput: bool = False,
                epochs: int = 5):
    losses = []
    metrics = {'train_accuracy': [],
               'valid_accuracy': [],
                'train_f1': [],
                'valid_f1': []}
    
    best_valid_loss = np.inf

    for n_epoch in range(epochs):

        train_losses = []
        train_preds = []
        train_targets = []
        valid_losses = []
        valid_preds = []
        valid_targets = []

        progress_bar = tqdm(total=len(train_loader.dataset), desc=f'Epoch {n_epoch + 1} of {epochs}')

        model.train()
        

        for batch_index, (x, y) in enumerate(train_loader):
            
            if gpu:
                x = x.to(device)
                y = y.to(device)
            
            x = x.view(x.shape[0], -1)

            optimizer.zero_grad()

            preds = model(x)

            loss = criterion(preds, y)

            loss.backward()

            optimizer.step()
            
            if batch_index % 100 == 0:
                scheduler.step()
#                 print(scheduler.get_last_lr()) # debug

            train_losses.append(loss.item())
            losses.append(loss.item())

            train_preds.append(torch.argmax(preds, dim=1).view(-1, 1).numpy())
            train_targets.append(y.numpy())

            progress_bar.set_postfix(train_loss=np.mean(losses[-500:]))

            progress_bar.update(x.shape[0])

        progress_bar.close()

        model.eval()

        for x, y in valid_loader:
            x = x.view(x.shape[0], -1)

            with torch.no_grad():
                preds = model(x)

            valid_preds.append(torch.argmax(preds, dim=1).view(-1, 1).numpy())
            valid_targets.append(y.numpy())

            loss = criterion(preds, y)

            valid_losses.append(loss.item())

        print(50 * '-')
        print(f'Epoch {n_epoch + 1} results')
        print(f'Mean train loss: {np.mean(train_losses):.3f}, Mean valid loss: {np.mean(valid_losses):.3f}')

        train_accuracy = accuracy_score(np.concatenate(train_targets),
                                        np.concatenate(train_preds))
        valid_accuracy = accuracy_score(np.concatenate(valid_targets),
                                        np.concatenate(valid_preds))
        
        train_f1 = f1_score(np.concatenate(train_targets),
                            np.concatenate(train_preds),
                            average='weighted')
                
        valid_f1 = f1_score(np.concatenate(valid_targets),
                            np.concatenate(valid_preds),
                            average='weighted')

        metrics['train_accuracy'].append(train_accuracy)
        metrics['valid_accuracy'].append(valid_accuracy)
        metrics['train_f1'].append(train_f1)
        metrics['valid_f1'].append(valid_f1)

        print(f'Train accuracy: {train_accuracy:.3f}, valid accuracy: {valid_accuracy:.3f}')
        print(f'Train weighted F1: {train_f1:.3f}, valid weighted F1: {valid_f1:.3f}')
        print(50 * '-')

    return model, losses, metrics, np.concatenate(valid_preds), np.concatenate(valid_targets)

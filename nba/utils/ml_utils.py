import os
import sys
import torch
import torch.nn as nn
from nba.exception.exception import NbaException
from nba.logging.logger import logging
from tqdm import tqdm
import numpy as np

def train_one_epoch_mlp(model, loader, optimizer, device):
    try: 
        model.train()
        running_loss = 0.0
        criterion = nn.MSELoss(reduction='mean')

        for batch in tqdm(loader, desc="Training", leave=False):
            home_features = batch['home_features'].to(device)
            away_features = batch['away_features'].to(device)
            home_auxiliary_target = batch['home_auxiliary_target'].to(device)
            away_auxiliary_target = batch['away_auxiliary_target'].to(device)
            score_home = batch['score_home'].to(device)
            score_away = batch['score_away'].to(device)
            prob_home = batch['prob_home'].to(device)
            prob_away = batch['prob_away'].to(device)

            optimizer.zero_grad()
            # Forward pass
            outputs = model(home_features, away_features)
            
            target = prob_home.unsqueeze(1)
            loss = criterion(outputs, target)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * home_features.size(0)  # Accumulate loss
        
        epoch_loss = running_loss / len(loader.dataset)
        return epoch_loss

    except Exception as e:
        raise NbaException(e, sys)


# def evaluate_mlp(model, loader, device):
#     try: 
#         model.eval()
#         correct = 0.0

#         with torch.no_grad():
#             for batch in tqdm(loader, desc="Evaluating", leave=False):
#                 home_features = batch['home_features'].to(device)
#                 away_features = batch['away_features'].to(device)
#                 home_auxiliary_target = batch['home_auxiliary_target'].to(device)
#                 away_auxiliary_target = batch['away_auxiliary_target'].to(device)
#                 score_home = batch['score_home'].to(device)
#                 score_away = batch['score_away'].to(device)
#                 prob_home = batch['prob_home'].to(device)
#                 prob_away = batch['prob_away'].to(device)

#                 # Forward pass
#                 outputs = model(home_features, away_features)
                
#                 predictions = (outputs <= 0.5).long().squeeze()
#                 targets = torch.argmax(torch.stack([score_home, score_away], dim=1), dim=1)

#                 correct += (predictions == targets).sum().item()
#             acc = correct / len(loader.dataset)           
#             return acc
#     except Exception as e:
#         raise NbaException(e, sys)

def evaluate_mlp(model, loader, device):
    try: 
        model.eval()
        correct = 0.0
        cnt = 0 
        res = [] 
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                home_features = batch['home_features'].to(device)
                away_features = batch['away_features'].to(device)
                home_auxiliary_target = batch['home_auxiliary_target'].to(device)
                away_auxiliary_target = batch['away_auxiliary_target'].to(device)
                score_home = batch['score_home'].to(device)
                score_away = batch['score_away'].to(device)
                prob_home = batch['prob_home'].to(device)
                prob_away = batch['prob_away'].to(device)

                # Forward pass
                outputs = model(home_features, away_features)
                

                for i in range(outputs.size(0)):
                    res.append((outputs[i].item(), int(score_home[i] < score_away[i])))
                    cnt += 1 

            stats = evaluate_predictions(res, top_percents=(10,20,50))


            return stats

    except Exception as e:
        raise NbaException(e, sys)

def evaluate_predictions(data, top_percents=(10, 20, 50)):
    """
    data: list of [prob_home_win, true_label]
          true_label: 0=home win, 1=away win
    top_percents: tuple of percentages of predictions to evaluate

    Returns: dict with overall accuracy and per-top-k accuracy
    """
    arr = np.array(data)
    probs = arr[:, 0].astype(float)
    y_true = arr[:, 1].astype(int)

    # predictions: 0 if prob>0.5 else 1
    y_pred = (probs <= 0.5).astype(int)

    # overall accuracy
    overall_acc = (y_pred == y_true).mean()

    # confidence measure = |p - 0.5|
    confidence = np.abs(probs - 0.5)

    # sort indices by confidence descending
    order = np.argsort(-confidence)

    stats = {"overall": float(overall_acc)}

    n = len(probs)
    for perc in top_percents:
        k = max(1, int(n * perc / 100))
        idx = order[:k]
        acc = (y_pred[idx] == y_true[idx]).mean()
        stats[f"top_{perc}_preds"] = float(acc)

    return stats
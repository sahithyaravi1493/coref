import pandas as pd
from ..utils import load_pkl_dump, load_pickle
from itertools import combinations
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from sklearn.utils import shuffle
from collections import Counter
from evaluator import Evaluation
import wandb
import os
from config_sentence_based import *
from create_pairwise_data import *

# Weights and biases logging config
# wandb 
os.environ["WANDB_SILENT"] = "true"
config={"epochs": 20, "batch_size": 1024, "lr":1e-3}

wandb.init(
  project="coref-pairwise",
  notes="this is the hand-curated dataset with labels set to 1 if two embeddings are the same",
#   tags=["hand-curated", "train"],
  config=config,
)
wandb.run.name = f'{EMB_TYPE}-embeddings-test-balanced'

# choose embedding 
if EMB_TYPE == 'rgcn':
    # RGCN FINAL LAYER
    INPUT_LAYER = 51*200 #200
    semb_train = r_train
    semb_val = r_val

elif EMB_TYPE == 'sent':
    # Initial COMET sentence embeddings
    INPUT_LAYER = 1024
    semb_train = s_train
    semb_val = s_val
else:
    # Initial node embeddings - original COMET csk inferences +sentences
    INPUT_LAYER = 51*1024 
    semb_train = n_train
    semb_val = n_val


def batch_saved_embeddings(batch_ids, embedding):
    """
    Returns embeddings of a batch
    @param batch_ids: keys of the batch
    @param config:  configuration variables
    @param embedding: dict of embeddings
    @return:
    """
    batch_embeddings = []
    for ind in batch_ids:
        if EMB_TYPE == "rgcn":
            out = embedding[ind]
        else:
            # sentence embeddings are stored as a dataframe, so use .loc
            out = embedding.loc[ind][0]
        batch_embeddings.append(out)
    return np.array(batch_embeddings)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.uniform_(m.bias)

class SimplePairWiseClassifier(nn.Module):
    def __init__(self):
        super(SimplePairWiseClassifier, self).__init__()
        self.input_layer = INPUT_LAYER
        self.input_layer *= 3
        self.hidden_layer = 1024
        self.pairwise_mlp = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.input_layer, self.hidden_layer*2),
            nn.ReLU(),
            nn.Linear(self.hidden_layer*2, self.hidden_layer),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(self.hidden_layer, 1),
        )
        self.pairwise_mlp.apply(init_weights)

    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))
        #return self.pairwise_mlp(torch.cat((first, second), dim=1))



def hand_curated_labels(df_clusters):
    # Create a custom dataset where label =1 if sentences are the same, and 0 if they are different
    s1_id = []
    s2_id = []
    labels = []
    clus_ids = []
    unique_cluster_ids = df_clusters["cluster_id"].unique()
    first, second = zip(*list(combinations(range(len(unique_cluster_ids)), 2)))
    for i,j in zip(first, second):
        if i != j:
            # print(unique_cluster_ids[i], unique_cluster_ids[j])
            df1 = df_clusters[df_clusters["cluster_id"] == unique_cluster_ids[i]]
            df2 = df_clusters[df_clusters["cluster_id"] == unique_cluster_ids[j]]

            id1, id2 = df1["combined_id"].values[0], df2["combined_id"].values[0]
            if id1 == id2:
                labels.append(1)
                clus_ids.append(i + j)
                s1_id.append(id1)
                s2_id.append(id2)
            else:
                if Counter(labels)[0] < 100:
                    labels.append(0)
                    clus_ids.append(i + j)
                    s1_id.append(id1)
                    s2_id.append(id2)

    dataset = pd.DataFrame()
    dataset["cid"] = clus_ids
    dataset["s1_id"] = s1_id
    dataset["s2_id"] = s1_id
    dataset["label"] = labels
    return dataset

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("### USING GPU:0")
        device = torch.device('cuda')  
    else:
        print("### USING CPU")
        device = 'cpu'

    t,v = get_pairwise_data()
    print("Label distribution of train:", t['label'].value_counts())
    print("Label distribution of val", v['label'].value_counts())

    # Init model
    model = SimplePairWiseClassifier().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([WEIGHT]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Train model
    n_epochs = config["epochs"]
    model.train()
    batch_size = config["batch_size"]
 
    for epoch in range(n_epochs):
        accumulate_loss = 0
        first_ids = t['s1_id'].values
        second_ids = t['s2_id'].values
        labels = t['label'].values
        idx = shuffle(list(range(len(first_ids))), random_state=50)
        for i in range(0, len(first_ids), batch_size):
            indices = idx[i:i+batch_size]
            
            batch_first_ids, batch_second_ids =  first_ids[indices], second_ids[indices]
            batch_labels = torch.tensor(labels[indices]).float().cuda()
            batch_first = batch_saved_embeddings(batch_first_ids, semb_train)
            batch_second = batch_saved_embeddings(batch_second_ids, semb_train)
            graph1 = torch.tensor(batch_first).float().cuda()
            graph2 = torch.tensor(batch_second).float().cuda()
            optimizer.zero_grad()
            y_pred = model(graph1,graph2 )
            # Compute Loss
            #print("labels, preds", batch_labels.shape, y_pred.shape)
            loss = criterion(y_pred, batch_labels.reshape(-1,1))
            accumulate_loss += loss.item()
            # Backward pass
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        print('Epoch {}: Train loss: {}'.format(epoch, accumulate_loss))
        wandb.log({"train loss":accumulate_loss})



        model.eval()
        accumul_val_loss = 0
        with torch.no_grad():
            first_ids = v['s1_id'].values
            second_ids = v['s2_id'].values
            labels = v['label'].values
            idx = list(range(len(first_ids)))
            
            all_scores = []
            all_labels = []
            for i in range(0, len(first_ids), batch_size):
                indices = idx[i:i+batch_size]
                batch_first_ids, batch_second_ids =  first_ids[indices], second_ids[indices]
                batch_labels = torch.tensor(labels[indices]).float().cuda()
                batch_first = batch_saved_embeddings(batch_first_ids, semb_val)
                batch_second = batch_saved_embeddings(batch_second_ids, semb_val)
                graph1 = torch.tensor(batch_first).float().cuda()
                graph2 = torch.tensor(batch_second).float().cuda()
                scores = model(graph1,graph2 )
                loss = criterion(scores, batch_labels.reshape(-1,1))
                accumul_val_loss += loss.item()
                all_scores.extend(scores.squeeze())
                all_labels.extend(batch_labels.to(torch.int))
        
        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)
        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        strict_preds = (all_scores > 0).to(torch.int)
        wandb.log({"val loss": accumul_val_loss})
        print("Validation loss for this epoch: ", accumul_val_loss)
        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len(torch.nonzero(all_labels == 1)),
                                                                len(all_labels)))
        f1 = eval.get_f1()

        print('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                eval.get_precision(), f1))
        wandb.log({"f1": f1})
import pandas as pd
from utils import load_pkl_dump, load_pickle
from itertools import combinations
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import os
from sklearn.utils import shuffle
from collections import Counter
from evaluator import Evaluation
import wandb
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used

os.environ["WANDB_SILENT"] = "true"
config={"epochs": 25, "batch_size": 512, "lr":1e-5}

wandb.init(
  project="coref-pairwise",
  notes="this is the hand-curated dataset with labels set to 1 if two embeddings are the same",
  tags=["hand-curated", "train"],
  config=config,
)
wandb.run.name = 'rgcn-embeddings-test'
# Define all paths

cluster_paths = {
    'train': '/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_train.csv',
    'val':'/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_dev.csv'
    }

r_train = load_pkl_dump('/ubc/cs/research/nlp/sahiravi/coref/comet/rgcn_hidden_train')
r_val = load_pkl_dump('/ubc/cs/research/nlp/sahiravi/coref/comet/rgcn_hidden_val')
s_train = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/sentence_embeddings_train.pkl')
s_val = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/sentence_embeddings_val.pkl')
cs_train = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/expansion_embeddings_train.pkl')
cs_val = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/expansion_embeddings_val.pkl')

# Define params
EMB_TYPE = 'rgcn' #'sent'
if EMB_TYPE == 'rgcn':
    INPUT_LAYER = 51*200 #200
    semb_train = r_train
    semb_val = r_val
else:
    INPUT_LAYER = 1024
    semb_train = s_train
    semb_val = s_val
HAND_CURATED = False

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
            nn.Linear(self.input_layer, self.hidden_layer),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_layer, 512),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(512, 1),
        )
        self.pairwise_mlp.apply(init_weights)

    def forward(self, first, second):
        return self.pairwise_mlp(torch.cat((first, second, first * second), dim=1))
        #return self.pairwise_mlp(torch.cat((first, second), dim=1))

def create_labels(df_clusters, s):
    # Get dataframe containing cluster information
    cluster_pairs = []
    print(df_clusters[df_clusters["sentence_id_x"]!= df_clusters["sentence_id_y"]])
    print(df_clusters.columns)
    df_clusters.dropna(subset=['cluster_desc'], how='all', inplace=True)
    groups = df_clusters.groupby('cluster_id')
    clus_ids = []
    s1_id = []
    s2_id = []
    s1 = []
    s2 = []
    label = []
    for cluster_id, frame in groups:
        frame["combined_id"] = frame["doc_id_x"] + "_" + frame["sentence_id_x"].astype(str)

        # Stick to cluster sizes > 1

        if frame["combined_id"].nunique() > 1:
            corefering = frame["combined_id"].values
            first, second = zip(*list(combinations(range(len(corefering)), 2)))
            for i,j in zip(first, second):
                clus_ids.append(frame["cluster_id"].values[i])
                id1, id2 = frame["combined_id"].values[i], frame["combined_id"].values[j]
                s1_id.append(id1)
                s2_id.append(id2)
                # s1.append(s[id1].squeeze().astype(float))
                # s2.append(s[id2].squeeze().astype(float))
                label.append(1)

            


    unique_cluster_ids = df_clusters["cluster_id"].unique()
    first, second = zip(*list(combinations(range(len(unique_cluster_ids)), 2)))
    for i,j in zip(first, second):
        if i != j:
            # print(unique_cluster_ids[i], unique_cluster_ids[j])
            df1 = df_clusters[df_clusters["cluster_id"] == unique_cluster_ids[i]]
            df2 = df_clusters[df_clusters["cluster_id"] == unique_cluster_ids[j]]
            id1, id2 = df1["combined_id"].values[0], df2["combined_id"].values[0]
            # if id1 != id2:
            clus_ids.append(-1)
            
            s1_id.append(id1)
            s2_id.append(id2)
            # s1.append(s[id1].squeeze().astype(float))
            # s2.append(s[id2].squeeze().astype(float))
            label.append(0)
            

    dataset = pd.DataFrame()
    dataset["cid"] = clus_ids
    dataset["s1_id"] = s1_id
    dataset["s2_id"] = s1_id
    dataset["label"] = label
    # dataset["emb1"] = s1
    # dataset["emb2"] = s2
    # print(dataset["label"].value_counts())
    return dataset


def hand_curated_labels(df_clusters):
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
    
    # read clusters from train and val


    # if labels don't already exist, create them
    if HAND_CURATED:
        train = pd.read_csv(cluster_paths['train'])
        val = pd.read_csv(cluster_paths['val'])
        t = hand_curated_labels(train)
        v = hand_curated_labels(val)
        # v = t 
        # semb_val = semb_train
    else:
        if not os.path.exists('t.pkl'):
            train = pd.read_csv(cluster_paths['train'])
            val = pd.read_csv(cluster_paths['val'])
            t = create_labels(train, semb_train)
            v = create_labels(val, semb_val)
            t.to_pickle('t.pkl')
            v.to_pickle('v.pkl')
        else:
            t = pd.read_pickle('t.pkl')
            v = pd.read_pickle('v.pkl')
    
    print("val cns", t['label'].value_counts(), v['label'].value_counts())

    # Init model
    model = SimplePairWiseClassifier().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Train model
    n_epochs = config["epochs"]
    model.train()
    batch_size = config["batch_size"]
 
    for epoch in range(n_epochs):
        accumulate_loss = 0
        first_ids = t['s1_id'].values
        second_ids = t['s2_id'].values
        labels = t['label'].values
        idx = shuffle(list(range(len(first_ids))), random_state=20)
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
            loss = criterion(y_pred.squeeze(1), batch_labels)
            accumulate_loss += loss.item()
            # Backward pass
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        print('Epoch {}: Train loss: {}'.format(epoch, accumulate_loss))
        wandb.log({"loss":accumulate_loss})



        model.eval()
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
                loss = criterion(scores.squeeze(1), batch_labels)
                print("val loss", loss)
                all_scores.extend(scores.squeeze())
                all_labels.extend(batch_labels.to(torch.int))
        
        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)
        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        strict_preds = (all_scores > 0).to(torch.int)
        print('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        print('Number of positive pairs: {}/{}'.format(len(torch.nonzero(all_labels == 1)),
                                                                len(all_labels)))
        print('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                eval.get_precision(), eval.get_f1()))
        
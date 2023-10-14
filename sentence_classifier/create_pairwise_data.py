import pandas as pd
from itertools import combinations
from config_sentence_based import *

def create_labels(df_clusters, balance=False):
    # First find all corefering sentence pairs
    df_clusters.dropna(subset=['cluster_desc'], how='all', inplace=True)
    groups = df_clusters.groupby('cluster_id')
    clus_ids = []
    s1_id = []
    s2_id = []
    s1 = []
    s2 = []
    label = []
    seen = set()
    for cluster_id, frame in groups:
        frame["combined_id"] = frame["doc_id_x"] + "_" + frame["sentence_id_x"].astype(str)
        # Stick to cluster sizes > 1
        if frame["combined_id"].nunique() > 1:
            corefering = frame["combined_id"].values
            first, second = zip(*list(combinations(range(len(corefering)), 2)))
            for i,j in zip(first, second):
                clus_ids.append(frame["cluster_id"].values[i])
                id1, id2 = frame["combined_id"].values[i], frame["combined_id"].values[j]
                if (id1, id2) not in seen:
                    s1_id.append(id1)
                    s2_id.append(id2)
                    # s1.append(s[id1].squeeze().astype(float))
                    # s2.append(s[id2].squeeze().astype(float))
                    label.append(1)
                    seen.add((id1, id2))

    # Second find all non-corefering sentence pairs and add to lists
    conflicting = set()
    unique_cluster_ids = df_clusters["cluster_id"].unique()
    first, second = zip(*list(combinations(range(len(unique_cluster_ids)), 2)))
    for i,j in zip(first, second):
        if i != j:
            # print(unique_cluster_ids[i], unique_cluster_ids[j])
            df1 = df_clusters[df_clusters["cluster_id"] == unique_cluster_ids[i]]
            df2 = df_clusters[df_clusters["cluster_id"] == unique_cluster_ids[j]]
            min_size = min(len(df1), len(df2))
            if min_size > 1:
                f, s = zip(*list(combinations(range(min_size), 2)))
                for k,l in zip(f,s):
                    id1, id2 = df1["combined_id"].values[k], df2["combined_id"].values[l]
                    # if id1 != id2:
                    
                    if (id1, id2) not in seen:
                        clus_ids.append(-1)
                        s1_id.append(id1)
                        s2_id.append(id2)
                        # s1.append(s[id1].squeeze().astype(float))
                        # s2.append(s[id2].squeeze().astype(float))
                        label.append(0)
                        # break
                    else:
                        conflicting.add((id1, id2))
            

    dataset = pd.DataFrame()
    dataset["cid"] = clus_ids
    dataset["s1_id"] = s1_id
    dataset["s2_id"] = s2_id
    dataset["label"] = label
    print("CONFLICTS", len(conflicting))
    if balance:
        g = dataset.groupby('label')
        g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
        return g
    return dataset


def get_pairwise_data(balance=False):
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
            t = create_labels(train, balance=balance)
            v = create_labels(val)
            t.to_pickle('t.pkl')
            v.to_pickle('v.pkl')
        else:
            t = pd.read_pickle('t.pkl')
            v = pd.read_pickle('v.pkl')
    return t, v
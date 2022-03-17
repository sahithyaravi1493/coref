from config_sentence_based import *
from relational_graph_embeddings import csk_vectors
from create_pairwise_data import *

from collections import defaultdict
from distutils import core
import pandas as pd
import plotly
from torch import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

from numpy import dot
from numpy.linalg import norm
import numpy as np

from itertools import combinations
from scipy.spatial import distance

import plotly.express as px
import plotly.figure_factory as ff




# semb = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/sentence_embeddings_val.pkl')
# csemb = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/expansion_embeddings_val.pkl')

# cluster_path = '/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_dev.csv'
# embedding_path = '/ubc/cs/research/nlp/sahiravi/coref/comet/rgcn_init_val'
# embs = load_pkl_dump(embedding_path)

def process_csk_embs(batch_node_embeddings, batch_sen_doc_ids):
    # batch_node_embeddings = batch_node_embeddings.cpu().detach().numpy()
    batch_node_embeddings = batch_node_embeddings.reshape(batch_node_embeddings.shape[0], 1, batch_node_embeddings.shape[1])
    outputs = []
    n_relations = 10
    n_beams = 5
    batch_size = len(batch_sen_doc_ids)
    csk_ind = 0
    all_emb = None
    for i,sent in enumerate(batch_sen_doc_ids):
        for r in range(n_relations):
            for b in range(n_beams):
                cs_val = batch_node_embeddings[csk_ind]
                if all_emb is None:
                    all_emb = cs_val
                else:
                    all_emb = np.concatenate((all_emb, cs_val), axis=1)
                    csk_ind += 1
        outputs.append(all_emb)
    outputs = np.array(outputs)
    return outputs.reshape(len(batch_sen_doc_ids), outputs.shape[-1])


def cluster_pairs():

    # Get dataframe containing cluster information
    cluster_pairs = []
    df_clusters = pd.read_csv(cluster_paths['val'])
    print(df_clusters[df_clusters["sentence_id_x"]!= df_clusters["sentence_id_y"]])
    print(df_clusters.columns)
    df_clusters.dropna(subset=['cluster_desc'], how='all', inplace=True)
    groups = df_clusters.groupby('cluster_id')
    for cluster_id, frame in groups:
        frame["combined_id"] = frame["doc_id_x"] + "_" + frame["sentence_id_x"].astype(str)
        # Stick to cluster sizes > 1
        if frame["combined_id"].nunique() > 1:
            corefering = frame["combined_id"].values
            first, second = zip(*list(combinations(range(len(corefering)), 2)))
            for i,j in zip(first, second):
                if i!=j and corefering[i] != corefering[j]:
                    cluster_pairs.append((corefering[i], corefering[j]))
    
    return cluster_pairs


def non_cluster_pairs():
    cluster_pairs = []
    df_clusters = pd.read_csv(cluster_paths['val'])
    print(df_clusters.columns)
    df_clusters.dropna(subset=['cluster_desc'], how='all', inplace=True)
    df_clusters["combined_id"] = df_clusters["doc_id_x"] + "_" + df_clusters["sentence_id_x"].astype(str)
    groups = df_clusters.groupby('cluster_id')
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
                    id1 = df1["combined_id"].values[k]
                    id2 = df2["combined_id"].values[l]
                    if id1 != id2:
                        cluster_pairs.append((id1, id2))

    return cluster_pairs


def sims_csk(pairs):
    sims = []
    for x,y in pairs:
        x1 = csk_vectors(x, cs_val)
        x2 = csk_vectors(y, cs_val)
        v1 = process_csk_embs(x1, [x]).squeeze()
        v2 = process_csk_embs(x2, [y]).squeeze()
        
        sims.append( dot(v1, v2)/(norm(v1)*norm(v2)))
    return sims


def cosine_sims(pairs):
    sen_sims = []
    rgcn_sims = []
    v1s = []
    v2s = []
    for x,y in pairs:
        # # sent
        v1 = s_val.loc[x][0]
        v2 = s_val.loc[y][0]
        c1 = r_val[x].squeeze()
        c2 = r_val[y].squeeze()
        rgcn_sims.append( dot(c1, c2)/(norm(c1)*norm(c2)))
        sen_sims.append( dot(v1, v2)/(norm(v1)*norm(v2)))
        
    print(v1, v2)
    print("embedding size", r_val[x].shape)
    # sims = cosine_similarity(np.array(v1s), np.array(v2s))
    return sen_sims, rgcn_sims
        
if __name__ == '__main__':
    df_train, df_val = get_pairwise_data()
    df_val["label"] = df_val["label"].astype(int)
    corefering = df_val.loc[df_val["label"]==1]
    non_corefering = df_val.loc[df_val["label"]==0]
    print(corefering.head())
    print(non_corefering.head())
    
    sample_corefering_pairs = zip(list(corefering["s1_id"].values), list(corefering["s2_id"].values))
    sample_non_corefs = zip(list(non_corefering["s1_id"].values), list(non_corefering["s2_id"].values))
    sample_corefering_pairs = list(sample_corefering_pairs)
    sample_non_corefs = list(sample_non_corefs)
    print("Coref vs non coref size", len(sample_corefering_pairs), len(sample_non_corefs))



    #  df = pd.DataFrame()
    # df["corefering"] = c1 
    # df["non_corefering"] = c2
    # fig = px.ecdf(df, x=["corefering", "non_corefering"])
    # fig.update_layout(title_text='CDF CSK embeddings (corefering)',   xaxis_title="Cosine similarity")
    # fig.show()
    # # # cdf sentence embeddings
    # df = pd.DataFrame()
    # df["corefering"] = c1 
    # fig = px.ecdf(df, x=["corefering"])
    # fig.update_layout(title_text='CDF csk embeddings (corefering)',   xaxis_title="Cosine similarity")
    # fig.show()
    # df = pd.DataFrame()
    # df["non_corefering"] = c2
    # fig = px.ecdf(df, x=["non_corefering"])
    # fig.update_layout(title_text='CDF csk embedding (non-corefering)',   xaxis_title="Cosine similarity")
    # fig.show()

    sent1, r1= cosine_sims(sample_corefering_pairs)
    sent2, r2 = cosine_sims(sample_non_corefs)
    print("corefering pairs avg cosine sim...")
    print(sum(sent1)/len(sent1))
    print("non corefering pairs avg cosine sim...")
    print(sum(sent2)/len(sent2))

    # distribution plot of sentence embddings
    fig = ff.create_distplot([sent1, sent2], ['corefering pairs', 'non-corefering pairs'], bin_size=0.1)
    fig.update_layout(title_text='Cosine similarity of sentence embeddings(COMET)')
    fig.show()
    # distribution of rgcn embeddings
    fig = ff.create_distplot([r1, r2], ['corefering pairs', 'non-corefering pairs'], bin_size=0.1)
    fig.update_layout(title_text='Cosine similarity of rgcn embeddings(R-GCN)')
    fig.show()

    #CSK nodes only:
    # c1 = sims_csk(sample_corefering_pairs)
    # c2 = sims_csk(sample_non_corefs)
   
    # # # distribution plot of csk embddings
    # fig = ff.create_distplot([c1, c2], ['corefering pairs', 'non-corefering pairs'], bin_size=0.1)
    # fig.update_layout(title_text='Cosine similarity of csk embeddings(COMET)')
    # fig.show()
    # cdf sentence embeddings
    # df = pd.DataFrame()
    # df["corefering"] = sent1 
    # df["non_corefering"] = sent2[:len(sent1)]
    # fig = px.ecdf(df, x=["corefering", "non_corefering"])
    # fig.update_layout(title_text='CDF sentence embeddings (corefering)',   xaxis_title="Cosine similarity")
    # fig.show()
    # df = pd.DataFrame()
    # df["non_corefering"] = sent2
    # fig = px.ecdf(df, x=["non_corefering"])
    # fig.update_layout(title_text='CDF sentence embedding (non-corefering)',   xaxis_title="Cosine similarity")
    # fig.show()

    # cdf rgcn embeddings
    # df = pd.DataFrame()
    # df["corefering"] = r1 
    # df["non_corefering"] = r2[:len(sent1)]
    # fig = px.ecdf(df, x=["corefering", "non_corefering"])
    # fig.update_layout(title_text='CDF R-GCN embeddings (corefering)',   xaxis_title="Cosine similarity")
    # fig.show()
    # df = pd.DataFrame()
    # df["non_corefering"] = r2
    # fig = px.ecdf(df, x=["non_corefering"])
    # fig.update_layout(title_text='CDF R-GCN embeddings (non-corefering)',   xaxis_title="Cosine similarity")
    # fig.show()

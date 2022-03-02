import os
import pickle

import dgl
import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv.relgraphconv import RelGraphConv
from utils import save_pkl_dump, load_pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Embeddings saved using COMET (#TODO: move to parse args)
root_folder = '/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/'
save_folder = '/ubc/cs/research/nlp/sahiravi/coref/comet/'

# 1. Comet relations we used
comet_relations = [
    "AtLocation",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    "isAfter",
    "isBefore"
]

N_RELATIONS = len(comet_relations)
N_BEAMS = 5


class RelationalGCN(nn.Module):

    def __init__(self, hidden_features, out_features, rel_types=5, in_features=1024, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.layer1 = RelGraphConv(in_features, hidden_features, rel_types, regularizer='basis', num_bases=2)
        self.layer2 = RelGraphConv(hidden_features, out_features, rel_types, regularizer='basis', num_bases=2)
        self.device = device

    def forward(self, nx_graph, sentence_embeddings, csk_embeddings, relations):
        sentence_nodes = [n for n in nx_graph if nx_graph.nodes[n]['label'] == 'sentence']
        csk_nodes = [n for n in nx_graph if nx_graph.nodes[n]['label'] == 'csk']

        # embedding size = No of nodes * 1024
        embeddings = torch.zeros(nx_graph.number_of_nodes(), self.in_features).to(self.device)
        embeddings[sentence_nodes] = torch.tensor(sentence_embeddings).float().to(self.device)
        embeddings[csk_nodes] = torch.tensor(csk_embeddings).float().to(self.device)
        # print("embeddings shape ", embeddings.size())

        g = dgl.from_networkx(nx_graph)
        g = g.to(self.device)
        etype = torch.tensor(relations).to(self.device)

        hidden = F.relu(self.layer1(g, embeddings, etype))
        hidden = F.relu(self.layer2(g, hidden, etype))

        out = torch.cat([embeddings, hidden], -1)
        return embeddings, hidden, out


def csk_vectors(idx, cskb):
    """
    interleave csk vectors (useful when we have a batch_size > 1)

    @param idx:
    @param cskb:
    @return:
    """
    all_outs = []
    for rel in comet_relations:
        rel_out = cskb.loc[str(idx)][rel]
        all_outs.append(rel_out)
    all_outs = np.array(all_outs)
    # 10, 5, 1024 --->  50*1024
    shape_estimate = (len(all_outs) * all_outs[0].shape[0], all_outs[0].shape[-1])
    interleaved_array = np.hstack(all_outs).reshape(shape_estimate)
    return interleaved_array


def sentence_vectors(idx, sentence):
    """

    @param idx:
    @param sentence:
    @return:
    """
    vector = np.array(sentence.loc[str(idx)]["embedding"])
    return vector


# def process_graph_embeddings(node_embeddings, id):
#     """

#     @param node_embeddings: 
#     @param id:
#     @return:
#     """
#     node_embeddings = node_embeddings.cpu().detach().numpy()
#     print(node_embeddings.shape)
#     node_embeddings = node_embeddings.reshape(node_embeddings.shape[0], 1, node_embeddings.shape[1])
#     outputs = []
#     n_relations = N_RELATIONS
#     n_beams = 5
#     csk_ind = 1
#     all_emb = node_embeddings
#     for r in range(n_relations):
#         for b in range(n_beams):
#             csk_emb = node_embeddings[csk_ind]
#             all_emb = np.concatenate((all_emb, csk_emb), axis=1)
#             csk_ind += 1
#     print(all_emb.shape)
#     return all_emb


def initiate_graph_edges(sen_doc_id, csk_init, sentence_init):
    """
    Forms the networkx graph - sentence node connected to csk nodes
    @param sen_doc_id:
    @param csk_init:
    @param sentence_init:
    @return:
    """
    relations = []
    n_relations = N_RELATIONS
    n_beams = 5

    G = nx.MultiDiGraph()
    # 0th node is sentence
    G.add_node(0, label='sentence')
    # csk nodes start at 1
    csk_ind = 1
    for r in range(n_relations):
        for b in range(n_beams):
            # csk node
            G.add_node(csk_ind, label='csk')
            # join with sentence node
            G.add_edge(0, csk_ind)
            csk_ind += 1
            relations.append(r)

    csk_embedding = csk_vectors(sen_doc_id, csk_init)
    sentence_embedding = sentence_vectors(sen_doc_id, sentence_init)
    # print(csk_embedding.shape, sentence_embedding.shape, G.number_of_nodes)
    return G, sentence_embedding, csk_embedding, relations


def graph_gcn_vectors(model, idx, csk_init=None, sentence_init=None):
    """

    @param model: graph embedding model
    @param idx: index of the sentence
    @param csk_init: initial comet expansion embeddings
    @param sentence_init: initial sentence expansion embeddings
    @return:
    """
    # create the graph
    networkx_graph, sentence_vector, csk_vector, relations = initiate_graph_edges(idx, csk_init,
                                                                                  sentence_init)
    # embed the graph
    all_embeddings, hidden, out = model(networkx_graph, sentence_vector, csk_vector, relations)

    # all_embeddings = graph + relation embeddings
    # graph = process_graph_embeddings(hidden, idx)
    graph = hidden.flatten()
    reshaped_graph =  graph.reshape((1,-1))
    return reshaped_graph


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("### USING GPU:0")
        device = torch.device('cuda:0')  
    else:
        print("### USING CPU")
        device = 'cpu'

    # Initialize model
    EMBEDDING_DIM = 100
    graph_model = RelationalGCN(EMBEDDING_DIM, EMBEDDING_DIM, N_RELATIONS, device=device)
    graph_model = graph_model.to(device)

    # Path to save results
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # go through each split
    for split in ['train']:
        # load COMET embeddings from sentences and expansions
        cskb_emb = load_pickle(f"{root_folder}expansion_embeddings_{split}.pkl")
        sentence_emb = load_pickle(f"{root_folder}sentence_embeddings_{split}.pkl")

        # output graph embeddings gets collected here
        vectors = {}
        for ind, row in sentence_emb.iterrows():
            # pass each sentence, convert to graph
            v = graph_gcn_vectors(graph_model, ind, cskb_emb, sentence_emb)
            vectors[ind] = v
            

        print(f"Done with split {split}")
        
        save_pkl_dump(f"{save_folder}rgcn_hidden_{split}", vectors)

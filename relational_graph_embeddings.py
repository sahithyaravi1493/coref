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

    def forward(self, nx_graph, csk_embeddings, sentence_embeddings, relations):
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


def csk_vectors(ids, cskb):
    """
    interleave csk vectors (useful when we have a batch_size > 1)

    @param ids:
    @param cskb:
    @return:
    """
    all_outs = []
    for rel in comet_relations:
        rel_out = np.concatenate([cskb.loc[str(k)][rel] for k in ids]).squeeze()
        all_outs.append(rel_out)

    shape_estimate = (len(all_outs) * all_outs[0].shape[0], all_outs[0].shape[-1])
    interleaved_array = np.hstack(all_outs).reshape(shape_estimate)
    return interleaved_array


def sentence_vectors(ids, sentence):
    """

    @param ids:
    @param sentence:
    @return:
    """
    vector = np.array([sentence.loc[k]["embedding"] for k in ids])
    return vector


def process_graph_embeddings(batch_node_embeddings, batch_sen_doc_ids):
    """

    @param batch_node_embeddings:
    @param batch_sen_doc_ids:
    @return:
    """
    batch_node_embeddings = batch_node_embeddings.cpu().detach().numpy()
    batch_node_embeddings = batch_node_embeddings.reshape(batch_node_embeddings.shape[0], 1,
                                                          batch_node_embeddings.shape[1])
    outputs = []
    n_relations = N_RELATIONS
    n_beams = 5
    batch_size = len(batch_sen_doc_ids)
    csk_ind = len(batch_sen_doc_ids)
    # this loop is executed only once when batch_size =1
    for i, sent in enumerate(batch_sen_doc_ids):
        all_emb = batch_node_embeddings[i]
        for r in range(n_relations):
            for b in range(n_beams):
                csk_emb = batch_node_embeddings[csk_ind]
                all_emb = np.concatenate((all_emb, csk_emb), axis=1)
                csk_ind += 1
        outputs.append(all_emb)
    outputs = np.array(outputs)
    return outputs.reshape(len(batch_sen_doc_ids), outputs.shape[-1])


def initiate_graph_edges(batch_sen_doc_ids, csk_init, sentence_init):
    """
    Forms the networkx graph - sentence node connected to csk nodes
    @param batch_sen_doc_ids:
    @param csk_init:
    @param sentence_init:
    @return:
    """
    relations = []
    n_relations = N_RELATIONS
    n_beams = 5

    G = nx.MultiDiGraph()
    csk_ind = len(batch_sen_doc_ids)
    for i, sent in enumerate(batch_sen_doc_ids):
        # sentence node
        G.add_node(i, label='sentence')
        for r in range(n_relations):
            for b in range(n_beams):
                # csk node
                G.add_node(csk_ind, label='csk')
                G.add_edge(i, csk_ind)
                csk_ind += 1
                relations.append(r)

        csk_embeddings = csk_vectors(batch_sen_doc_ids, csk_init)
        sentence_embeddings = sentence_vectors(batch_sen_doc_ids, sentence_init)
        return G, sentence_embeddings, csk_embeddings, relations


def graph_gcn_vectors(model, batch_ids, csk_init=None, sentence_init=None):
    """

    @param model: graph embedding model
    @param batch_ids: ids from the initial embeddings which we want to embed as one graph
    @param csk_init: initial comet expansion embeddings
    @param sentence_init: initial sentence expansion embeddings
    @return:
    """
    # create the graph
    networkx_graph, sentence_vector, csk_vector, relations = initiate_graph_edges(batch_ids, csk_init,
                                                                                  sentence_init)
    # embed the graph
    all_embeddings, hidden, out = model(networkx_graph, sentence_vector, csk_vector, relations)

    # all_embeddings = graph + relation embeddings
    graph = process_graph_embeddings(hidden, batch_ids)
    return graph


if __name__ == '__main__':
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'
    graph_model = RelationalGCN(100, 100, N_RELATIONS, device=device)
    graph_model = graph_model.to(device)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for split in ['val', 'train']:
        cskb_emb = load_pickle(f"{root_folder}expansion_embeddings_{split}.pkl")
        sentence_emb = load_pickle(f"{root_folder}sentence_embeddings_{split}.pkl")
        vectors = {}

        for id, row in sentence_emb.iterrows():
            # batch size here is 1 as we want to have each graph embedding separate
            # note that we can pass array of ids and embed them together to a graph
            v = graph_gcn_vectors(graph_model, [id], cskb_emb, sentence_emb).squeeze()
            vectors[id] = v

        print(f"Done with split {split}")
        save_pkl_dump(f"{save_folder}rgcn_hidden_{split}", vectors)

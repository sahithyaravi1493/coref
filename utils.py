import collections
import logging
import os
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
import smtplib
import torch.optim as optim
from transformers import AdamW, get_linear_schedule_with_warmup
import json
from datetime import datetime
import pickle
from corpus import Corpus
from tqdm import tqdm
from models import SimpleFusionLayer
import plotly.express as px
import plotly.figure_factory as ff
import hickle as hkl

def create_corpus(config, tokenizer, split_name, is_training=True):
    docs_path = os.path.join(config.data_folder, split_name + '.json')
    mentions_path = os.path.join(config.data_folder,
                                 split_name + '_{}.json'.format(config.mention_type))
    with open(docs_path, 'r') as f:
        documents = json.load(f)

    mentions = []
    if is_training or config.use_gold_mentions:
        with open(mentions_path, 'r') as f:
            mentions = json.load(f)

    predicted_topics = None
    if not is_training and config.use_predicted_topics:
        with open(config.predicted_topics_path, 'rb') as f:
            predicted_topics = pickle.load(f)

    logging.info('Split - {}'.format(split_name))

    return Corpus(documents, tokenizer, config.segment_window, mentions, subtopic=config.subtopic, predicted_topics=predicted_topics)


def create_logger(config, create_file=True):
    os.makedirs(config["log_path"], exist_ok=True)

    logging.basicConfig(
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            # logging.FileHandler(os.path.join(config.log_path, "test.log")),
            logging.FileHandler(os.path.join(config.log_path, '{}.log'.format(
                datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('simple_example')
    logger.propagate = True

    return logger

# def create_logger(config, create_file=True):
#     logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S')
#     logger = logging.getLogger('simple_example')
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#
#     c_handler = logging.StreamHandler()
#     c_handler.setLevel(logging.INFO)
#     c_handler.setFormatter(formatter)
#     logger.addHandler(c_handler)
#
#     if create_file:
#         if not os.path.exists(config.log_path):
#             os.makedirs(config.log_path)
#
#         f_handler = logging.FileHandler(
#             os.path.join(config.log_path,'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))), mode='w')
#         f_handler.setLevel(logging.INFO)
#         f_handler.setFormatter(formatter)
#         logger.addHandler(f_handler)
#
#     logger.propagate = False
#
#     return logger


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fix_seed(config):
    torch.manual_seed(config.random_seed)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)


def get_loss_function(config):
    if config.loss == 'hinge':
        return torch.nn.HingeEmbeddingLoss()
    else:
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_class_weight]).cuda())


def get_optimizer(config, models):
    parameters = []
    for model in models:
        parameters += list(model.parameters())

    if config.optimizer == "adam":
        return optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    elif config.optimizer == "adamw":
        return AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay, eps=config.adam_epsilon)
    else:
        return optim.SGD(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def get_scheduler(optimizer, total_steps):
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_to_dic(dic, key, val):
    if key not in dic:
        dic[key] = []
    dic[key].append(val)


def send_email(user, pwd, recipient, subject, body):

    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")


def align_ecb_bert_tokens(ecb_tokens, bert_tokens):
    bert_to_ecb_ids = []
    relative_char_pointer = 0
    ecb_token = None
    ecb_token_id = None

    for bert_token in bert_tokens:
        if relative_char_pointer == 0:
            ecb_token_id, ecb_token, _, _ = ecb_tokens.pop(0)

        bert_token = bert_token.replace("##", "")
        if bert_token == ecb_token:
            bert_to_ecb_ids.append(ecb_token_id)
            relative_char_pointer = 0
        elif ecb_token.find(bert_token) == 0:
            bert_to_ecb_ids.append(ecb_token_id)
            relative_char_pointer = len(bert_token)
            ecb_token = ecb_token[relative_char_pointer:]
        else:
            print("When bert token is longer?")
            raise ValueError((bert_token, ecb_token))

    return bert_to_ecb_ids


def save_pkl_dump(filename, dictionary):
    with open(f'{filename}.pickle', 'wb') as handle:
        pickle.dump(dictionary, handle, protocol=2)


def load_pkl_dump(filename, ext='pkl'):
    if ext == 'hkl':
        a = hkl.load(f'{filename}.hkl')
        return a

    with open(f'{filename}.pickle', 'rb') as handle:
        a = pickle.load(handle)
    
    return a


def load_pickle(filepath):
    df = pd.read_pickle(filepath)
    return df


def load_stored_embeddings(config, split):
    if config.embedding_type == "rgcn" or config.embedding_type == "node":
        embedding = load_pkl_dump(f"{config.stored_embeddings_path}_{split}")
        return embedding
    else:
        embedding = load_pickle(f"{config.stored_embeddings_path}_{split}.pkl")
        return embedding


def batch_saved_embeddings(batch_ids, config, embedding):
    """

    @param batch_ids: keys of the batch
    @param config:  configuration variables
    @param embedding: dict of embeddings
    @return:
    """
    batch_embeddings = []
    for ind in batch_ids:
        if config.embedding_type == "rgcn" or config.embedding_type == "node":
            out = embedding[ind]
        else:
            # sentence embeddings are stored as a dataframe, so use .loc
            out = embedding.loc[ind][0]
        batch_embeddings.append([out])
        # print(out.shape)
    return np.array(batch_embeddings).reshape(len(batch_embeddings), -1)


def plot_this_batch(g1, g2, span1, span2, c1, c2, batch_labels):

    # Plot cosine similarity of embeddings g1 and g2
    cos = nn.CosineSimilarity(dim=1, eps=1e-8)
    cosine_similarities = cos(g1, g2).cpu().detach().numpy()
    batch_labels = batch_labels.cpu().detach().numpy()
    pos_indices = np.where(batch_labels == 1)[0]
    neg_indices = np.where(batch_labels == 0)[0]
    # print(len(pos_indices), len(neg_indices))
    if len(pos_indices) > 1 and len(neg_indices) > 1:
        fig = ff.create_distplot([cosine_similarities[pos_indices], cosine_similarities[neg_indices]], [
                                 'corefering pairs', 'non-corefering pairs'], bin_size=0.01)
        fig.update_layout(
            title_text='Cosine similarity of span start-end embeddings')
        fig.show()

    # Find how many corefering pairs have cosine sim less than 0.9
    less = (cosine_similarities[pos_indices] > 0.9).sum()
    count = len(cosine_similarities[pos_indices])
    print("Corefering pairs with similarity more than 0.9", less/count)

    # Find how many non-corefering pairs have cosine sim greater than 0.9
    great = (cosine_similarities[neg_indices] > 0.9).sum()
    count = len(cosine_similarities[neg_indices])
    print(f"Non-corefering pairs with cosine sim greater than 0.9 {count} {great/count}")

def load_json(filepath):
    with open(filepath, 'r') as fp:
        file = json.loads(fp.read())
    return file


def save_json(filename, data):
    with open(filename, 'w') as fpp:
        json.dump(data, fpp)


def final_vectors(first_batch_ids, second_batch_ids, config, span1, span2, embeddings, e1, e2):
    """
    if include_graph is set to false, returns the span embeddings
    if include_graph is set to true, returns the graph and/ span embeddings

    @param first_batch_ids: keys of first batch
    @param second_batch_ids: keys of second batch
    @param config: configuration variables
    @param span1: span1 embeddings
    @param span2: span2 embeddings
    @param embeddings: dict with all knowledge embeddings, we will look up the ids in this dict
    @return:
    """
    device = span1.device
    if not config.include_graph and not config.include_text:
        # if graph is not included, just use spans
        return span1, span2

    elif config.include_text:
        
        e1 = e1.reshape(len(first_batch_ids), -1)
        e2 = e2.reshape(len(second_batch_ids), -1)
        # print("PAIRWISE SHAPE", e1.size(), e2.size())

        if config.exclude_span_repr:
            g1_new, g2_new = e1, e2
        else:
            # Concatenate span + expansions
            if config.fusion == "concat":
                g1_new = torch.cat((span1, e1 ), axis=1)
                g2_new = torch.cat((span2, e2), axis=1)
            else:
                fusion_model = SimpleFusionLayer(config).to(device)
                # fusion_model.train()
                g1_new, g2_new = fusion_model(span1, e1), fusion_model(span2, e2)
                # print("Fusion", g1_new.shape)

    else:
        # if graph is included, load the saved embeddings for this batch
        graph1 = batch_saved_embeddings(first_batch_ids, config, embeddings)
        graph2 = batch_saved_embeddings(second_batch_ids, config, embeddings)
        graph1 = torch.tensor(graph1).float().cuda()
        graph2 = torch.tensor(graph2).float().cuda()

        if config.exclude_span_repr:
            # if this is set to true, we exclude spans entirely and only use graph
            g1_new, g2_new = graph1, graph2
        else:
            # Concatenate span + graph

            g1_new = torch.cat((span1, graph1), axis=1)
            g2_new = torch.cat((span2, graph2), axis=1)
    # print(g1_new.shape)
    return g1_new, g2_new


def get_span_specific_embeddings(topic_spans, span_repr, all_expansions, all_expansion_embeddings, span_embeddings, config):
   
    span_start_end_embeddings = topic_spans.start_end_embeddings
    combined_ids = topic_spans.combined_ids
    events = topic_spans.span_texts
    fine_grained_expansions = []
    csk_start_ends = []
    csk_widths = []
    csk_continuous = []
    misses = 0
    # print("Span specific embeddings calculation", span_embeddings.size(), len(combined_ids))
    if config.mode == "gpt3":
        for i in (range(len(combined_ids))):
            cid = combined_ids[i]
            event = events[i].strip()
            # print("EVent", cid, event, topic_spans.width[i])
            key = (cid, event)
            # The vector specific to this event
            se, cont, width = torch.zeros(2, 2048), torch.zeros(2, 1024), torch.tensor([80, 65])
        
            if key in all_expansion_embeddings['startend']:
                # Key value lookup of saved expansions
                se = torch.tensor(all_expansion_embeddings['startend'][key])
                cont = torch.tensor(all_expansion_embeddings['cont'][key])
                width = torch.tensor(all_expansion_embeddings['width'][key])
            else: 
                misses += 1
            # print(cont.size())
           
            selection = all_expansions.loc[all_expansions["combined_id"] == cid]
            selection = selection[all_expansions["event"] == event]
            if selection.empty:
                final_expansions = ""
            else:
                final_expansions = selection["predictions"].values[0]
            # print("CONT", cont.size())
            csk_start_ends.append(se)
            csk_widths.append(width)
            csk_continuous.append(cont)
            fine_grained_expansions.append(final_expansions)
    else:
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        for i in (range(len(combined_ids))):
            # Look up inferences of a particular sentence
            key = combined_ids[i]
            expansions = np.array(all_expansions[key])
            if config.attention_based:
                se = torch.tensor(all_expansion_embeddings['startend'][key]).cuda()
                cont = torch.tensor(all_expansion_embeddings['cont'][key]).cuda()
                width = torch.tensor(all_expansion_embeddings['width'][key]).cuda()
                with torch.no_grad():
                    candidate_tensors = span_repr(se, cont, width)
                span = span_embeddings[i].view(1,-1)
            else:
                candidate_tensors = torch.tensor(all_expansion_embeddings['startend'][key]).cuda()
                # find the top 5 expacnsion embeddings that are similar to the span
                span = span_start_end_embeddings[i].view(1, -1)
            # print(span.size(), candidate_tensors.size())
            
            # print(span)

            distances = cos(candidate_tensors, span)
            # if i == 0 or i == 10:
            #     print("min max:", torch.min(distances), torch.max(distances))
            values, indices = distances.topk(5)
            # print(values, indices)
            final_selection = candidate_tensors[indices].reshape(1, -1).squeeze()
            final_expansions = expansions[indices.cpu().detach().numpy()]
            csk_start_ends.append(final_selection)
            fine_grained_expansions.append(final_expansions)
    # print("WE MISSED", misses)
    # print(type(csk_continuous[0]), type(csk_widths[0]))
    return csk_start_ends, csk_continuous, csk_widths, fine_grained_expansions


def get_expansion_with_attention(span_repr, knowledge_embs, batch_first, batch_second, device):
    knowledge_start_end_embeddings, knowledge_continuous_embeddings, knowledge_width = knowledge_embs
    n_spans = len(knowledge_continuous_embeddings)
    n_relations = 2
    before_se, before_ce, before_w = [], [], []
    after_se, after_ce, after_w = [], [], []
    for i in range(n_spans):
        before_se.append(knowledge_start_end_embeddings[i][0])
        after_se.append(knowledge_start_end_embeddings[i][1])
        before_ce.append(knowledge_continuous_embeddings[i][0].reshape(-1, 1024).to(device))
        after_ce.append(knowledge_continuous_embeddings[i][1].reshape(-1, 1024).to(device))   
        before_w.append(knowledge_width[i][0])
        after_w.append(knowledge_width[i][1])
    
    before_se = torch.stack(before_se).to(device)
    after_se = torch.stack(after_se).to(device)
    before_w, after_w = torch.stack(before_w).to(device), torch.stack(after_w).to(device)
    # print(before_se.shape, before_w.shape)
    bef1 = span_repr(before_se[batch_first],
                       [before_ce[k] for k in batch_first], before_w[batch_first])
    bef2 = span_repr(before_se[batch_second],
                       [before_ce[k] for k in batch_second], before_w[batch_second])
    aft1 = span_repr(after_se[batch_first],
                       [after_ce[k] for k in batch_first], after_w[batch_first])
    aft2 = span_repr(after_se[batch_second],
                       [after_ce[k] for k in batch_second], after_w[batch_second])
    # print(bef1.shape, aft1.shape)
    e1 = torch.cat((bef1, aft1), axis=1)
    e2 = torch.cat((bef2, aft2), axis=1)
    return e1, e2
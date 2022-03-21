import sys, os
sys.path.append('/ubc/cs/research/nlp/sahiravi/coref')

from utils import load_pkl_dump, load_pickle
import wandb

# Where all clusters are saved
cluster_paths = {
    'train': '/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_train.csv',
    'val':'/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_dev.csv'
    }

# initial rgcn embeddings
n_train = load_pkl_dump('/ubc/cs/research/nlp/sahiravi/coref/comet/rgcn_init_train')
n_val = load_pkl_dump('/ubc/cs/research/nlp/sahiravi/coref/comet/rgcn_init_val')

# rgcn final embeddings
r_train = load_pkl_dump('/ubc/cs/research/nlp/sahiravi/coref/comet/rgcn_hidden_train')
r_val = load_pkl_dump('/ubc/cs/research/nlp/sahiravi/coref/comet/rgcn_hidden_val')

# sentence embeddings from COMET
s_train = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/sentence_embeddings_train.pkl')
s_val = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/sentence_embeddings_val.pkl')

# CSK embeddings from COMET
cs_train = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/expansion_embeddings_train.pkl')
cs_val = load_pickle('/ubc/cs/research/nlp/sahiravi/comet-atomic-2020/coref_expansion/expansion_embeddings_val.pkl')

# Define model parameters
config={"epochs": 10, "batch_size": 512, "lr":1e-4}

# weight used for increasing the importance of positive examples in the loss function
WEIGHT = 1

# Use balanced training
BALANCE = False

# Whether u want to use a hand-curated sample of 100, 100
HAND_CURATED = False

"""
'sent' = only sentence embeddings from COMET 1*1024
'rgcn' = rgcn final hidden layer  50 *200
'node' = initial node embeddings composed of csk nodes and sentence nodes 50*1024
"""
EMB_TYPE = 'rgcn'


# cuda devices
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"  # specify which GPU(s) to be used
 
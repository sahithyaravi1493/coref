import sys, os
import pandas as pd
sys.path.append('/ubc/cs/research/nlp/sahiravi/coref')

from utils import load_pkl_dump, load_pickle, load_json
import wandb


import logging
# Configure logging here
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.WARNING)
logger = logging.getLogger(__name__)

# all original clusters
cluster_paths = {
    'train': '/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_train.csv',
    'val':'/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_dev.csv',
    'test':'/ubc/cs/research/nlp/sahiravi/datasets/coref/filtered_ecb_corpus_clusters_test.csv'
    }


# all original sentences
sentences = {
        'train': pd.read_csv('/ubc/cs/research/nlp/sahiravi/datasets/coref/sentence_ecb_corpus_train.csv'),
        'val': pd.read_csv('/ubc/cs/research/nlp/sahiravi/datasets/coref/sentence_ecb_corpus_dev.csv'),
        'test': pd.read_csv('/ubc/cs/research/nlp/sahiravi/datasets/coref/sentence_ecb_corpus_test.csv')
    }

# all original expansions
expansions = {
        'train':load_json('coref_expansion/expansions_train.json'),
        'val': load_json('coref_expansion/expansions_val.json'),
        # 'test': '/ubc/cs/research/nlp/sahiravi/datasets/coref/sentence_ecb_corpus_test.csv'
    }
relation_map = load_json("/ubc/cs/research/nlp/sahiravi/coref/comet/relation_map_no_subject.json")
#load_json("/ubc/cs/research/nlp/sahiravi/coref/comet/relation_map.json")


atomic_relations = ["oEffect",
                    "oReact",
                    "oWant",
                    "xAttr",
                    "xEffect",
                    "xIntent",
                    "xNeed",
                    "xReact",
                    "xReason",
                    "xWant"]

excluded_relations = ["causes", "xReason", "isFilledBy", "HasPainCharacter", "HasPainIntensity" ] # exclude some relations that are nonsense
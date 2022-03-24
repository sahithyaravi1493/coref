from config_expansions import *
from transformers import AutoTokenizer, AutoModel
import torch
import pyhocon
from model_utils import pad_and_read_bert
from utils import save_pkl_dump
from tqdm import tqdm

config = pyhocon.ConfigFactory.parse_file('configs/config_pairwise.json')



if __name__ == '__main__':
    # print(config)
    if torch.cuda.is_available():
        print("### USING GPU:0")
        device = torch.device('cuda:0')  
    else:
        print("### USING CPU")
        device = 'cpu'
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_tokenizer'])
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    bert_model.eval()
    
    ids = bert_tokenizer.encode('granola bars')
    embeddings, len = pad_and_read_bert([ids], bert_model)
    print(embeddings.size())

    for split in ['train', 'val']:
        start_end_embeddings = {}
        continuous_embeddings = {}
        widths = {}
        all_expansions = load_json(f"comet/{split}_exp_sentences.json")
        for key, expansions in tqdm(all_expansions.items()):
            # Tokenize all inferences
            token_ids = []
            for inf in expansions:
                token_ids.append(bert_tokenizer.encode(inf))
            # Find all embeddings of the inferences and find start_end_embeddings
            embeddings, lengths = pad_and_read_bert(token_ids, bert_model)
            starts = embeddings[:, 0, :]
            ends = embeddings[:, -1, :]
            start_ends = torch.hstack((starts, ends))
            start_end_embeddings[key] = start_ends.cpu().detach().numpy()
            continuous_embeddings[key] = embeddings.cpu().detach().numpy()
            widths[key] = lengths
            #print(start_end_embeddings[key].shape, continuous_embeddings[key].shape, widths[key].shape)
            

        save_pkl_dump(f"comet/{split}_exp_startend", start_end_embeddings)
        save_pkl_dump(f"comet/{split}_exp_cont", continuous_embeddings)
        save_pkl_dump(f"comet/{split}_exp_widths", widths)

        



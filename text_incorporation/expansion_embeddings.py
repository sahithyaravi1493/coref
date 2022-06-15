from config_expansions import *
from transformers import AutoTokenizer, AutoModel
import torch
import pyhocon
from model_utils import pad_and_read_bert
from utils import save_pkl_dump
from tqdm import tqdm
import os
import hickle as hkl
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = pyhocon.ConfigFactory.parse_file('configs/config_pairwise.json')

def text_processing(inference):
    inference = inference.replace("After:", "")
    inference = inference.replace("Before:", "")
    inference = inference.replace("After,", "")
    inference = inference.replace("Before,", "")
    inference = inference.replace("\n", "")
    inference = inference.strip()
    return inference

if __name__ == '__main__':
    # Choose whether to embed GPT3 or COMET
    MODE = 'gpt3-individual'
    # Check GPU
    if torch.cuda.is_available():
        print("### USING GPU:0")
        device = 'cuda' 
    else:
        print("### USING CPU")
        device = 'cpu'

    # Load model based on configuration of pairwise scorer
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_tokenizer'])
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    bert_model.eval()
    ids = bert_tokenizer.encode('granola bars')
    embeddings, l = pad_and_read_bert([ids], bert_model)
    print(embeddings.size())
    start_end_embeddings = {}
    continuous_embeddings = {}
    widths = {}

    if MODE == "comet":
        for split in ['val']:
            all_expansions = load_json(f"comet/{split}_exp_sentences_ns.json")
            for key, expansions in tqdm(all_expansions.items()):
                # Tokenize all inferences
                token_ids = []
                for inf in expansions:
                    # print(inf)
                    token_ids.append(bert_tokenizer.encode(inf))
                # Find all embeddings of the inferences and find start_end_embeddings
                embeddings, lengths = pad_and_read_bert(token_ids, bert_model)
                starts = embeddings[:, 0, :]
                ends = embeddings[:, -1, :]
                start_ends = torch.hstack((starts, ends))
                start_end_embeddings[key] = start_ends.cpu().detach().numpy()
            # continuous_embeddings[key] = embeddings.cpu().detach().numpy()
                widths[key] = lengths
                torch.cuda.empty_cache()
                #print(start_end_embeddings[key].shape, continuous_embeddings[key].shape, widths[key].shape)
            
            hkl.dump(continuous_embeddings, f"comet/{split}_e_cont_ns.hkl", mode='w')
            save_pkl_dump(f"comet/{split}_e_startend", start_end_embeddings)
            save_pkl_dump(f"comet/{split}_e_cont", continuous_embeddings)
            save_pkl_dump(f"comet/{split}_e_widths", widths)
    elif MODE == "gpt3":
        
        for split in ['train', 'dev']:
            df = pd.read_csv(f'gpt3/output_{split}.csv')
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                key = (row["combined_id"], row["event"])
                all_expansions = row["predictions"]
                inferences = all_expansions.split("After:")
                for i in range(len(inferences)):
                    inferences[i] = text_processing(inferences[i])
                
                before_array = sorted([inf for inf in inferences[0].split(".") if len(inf.split())>3])
                after_array = sorted([inf for inf in inferences[1].split(".") if len(inf.split())>3])
                # print(after_array)
                before_inferences = ". ".join(before_array).lstrip(". ") + "."
                after_inferences = ". ".join(after_array).lstrip(". ")+ "."

                final_inferences = [before_inferences, after_inferences]
                token_ids = []
                for inf in final_inferences:
                    tokens = bert_tokenizer.encode(inf)
                    # print(len(tokens))
                    token_ids.append(tokens)
                # Find all embeddings of the inferences and find start_end_embeddings
                embeddings, lengths = pad_and_read_bert(token_ids, bert_model)
                print(embeddings.shape)
                starts = embeddings[:, 0, :]
                ends = embeddings[:, -1, :]
                start_ends = torch.hstack((starts, ends))
                start_end_embeddings[key] = start_ends.cpu().detach().numpy()
                # print(start_ends.shape)
                continuous_embeddings[key] = embeddings.cpu().detach().numpy()
                print(embeddings.shape)
                widths[key] = lengths
                torch.cuda.empty_cache()
                
            hkl.dump(start_end_embeddings, f"gpt3/{split}_e_startend_ns.hkl", mode='w')
            hkl.dump(widths, f"gpt3/{split}_e_widths_ns.hkl", mode='w')
            save_pkl_dump(f"gpt3/{split}_e_startend_ns", start_end_embeddings)
            save_pkl_dump(f"gpt3/{split}_e_widths_ns", widths)
            save_pkl_dump(f"gpt3/{split}_e_cont_ns", continuous_embeddings)


    elif MODE == "gpt3-individual":
        for split in ['train', 'dev']:
            df = pd.read_csv(f'gpt3/output_{split}.csv')
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                key = (row["combined_id"], row["event"])
                all_expansions = row["predictions"]
                inferences = all_expansions.split("After:")
                for i in range(len(inferences)):
                    inferences[i] = text_processing(inferences[i])
                
                before_array = sorted([inf+ "." for inf in inferences[0].split(".") if len(inf.split())>3])
                after_array = sorted([inf+ "." for inf in inferences[1].split(".") if len(inf.split())>3])
                # before_array = before_array + [" "]*(7-len(before_array))
                # after_array = after_array + [" "]*(7-len(after_array))
                final_array = before_array + after_array
                final_array.sort()
                token_ids = []
                for inf in final_array:
                    tokens = bert_tokenizer.encode(inf)
                    # print(len(tokens))
                    token_ids.append(tokens)
                # print(bert_tokenizer.pad_token_id)
                # Find all embeddings of the inferences and find start_end_embeddings
                
                embds, l = pad_and_read_bert(token_ids, bert_model)
               
                
                embeddings =  F.pad(embds, pad=(0, 0, 0, 0, 0, 16 - embds.shape[0]))
                lengths = np.pad(l, (0, 16-l.shape[0]), 'constant', constant_values=(0))
                # print(lengths.shape)

                starts = embeddings[:, 0, :]
                ends = embeddings[:, -1, :]
                start_ends = torch.hstack((starts, ends))
                start_end_embeddings[key] = start_ends.cpu().detach().numpy()
                # print(start_ends.shape)
                continuous_embeddings[key] = embeddings.cpu().detach().numpy()
                # print(embeddings.shape)
                widths[key] = lengths
                torch.cuda.empty_cache()
                
                
                
            # hkl.dump(start_end_embeddings, f"gpt3/{split}_e_startend_ns.hkl", mode='w')
            # hkl.dump(widths, f"gpt3/{split}_e_widths_ns.hkl", mode='w')
            save_pkl_dump(f"gpt3/{split}_e_startend_ind", start_end_embeddings)
            save_pkl_dump(f"gpt3/{split}_e_widths_ind", widths)
            save_pkl_dump(f"gpt3/{split}_e_cont_ind", continuous_embeddings)
            print(f"Done {split}")








    # for split in ['train','val']:
    #     print ("Saved embeddings, saving original sentences")
    #     original_sentences = dict(zip(list(sentences[split]["combined_id"].values), list(sentences[split]["sentence"].values)))
    #     for key, expansions in tqdm(original_sentences.items()):
    #         sentence = original_sentences[key]
    #         token_ids = [bert_tokenizer.encode(sentence)]
    #         # Find all embeddings of the inferences and find start_end_embeddings
    #         embeddings, lengths = pad_and_read_bert(token_ids, bert_model)
    #         starts = embeddings[:, 0, :]
    #         ends = embeddings[:, -1, :]
    #         start_ends = torch.hstack((starts, ends))
    #         start_end_embeddings[key] = start_ends.cpu().detach().numpy()
    #         continuous_embeddings[key] = embeddings.cpu().detach().numpy()
    #         widths[key] = lengths
    #     save_pkl_dump(f"comet/{split}_sent_startend", start_end_embeddings)
    #     save_pkl_dump(f"comet/{split}_sent_cont", continuous_embeddings)
    #     save_pkl_dump(f"comet/{split}_sent_widths", widths)

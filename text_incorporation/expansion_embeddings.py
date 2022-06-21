# import hickle as hkl
import numpy as np
import pyhocon
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config_expansions import *
from model_utils import pad_and_read_bert
from utils import save_pkl_dump

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = pyhocon.ConfigFactory.parse_file('configs/config_pairwise.json')
# Choose whether to embed GPT3 or COMET
commonsense_model = 'gpt3'



def comet_to_roberta_embeddings(bert_tokenizer, bert_model, comet_inferences_root="comet"):
    start_end_embeddings = {}
    continuous_embeddings = {}
    widths = {}
    for split in ['train', 'val']:
        all_expansions = load_json(f"{comet_inferences_root}/{split}_exp_sentences_ns.json")
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
            # print(start_end_embeddings[key].shape, continuous_embeddings[key].shape, widths[key].shape)

        hkl.dump(continuous_embeddings, f"comet/{split}_e_cont_ns.hkl", mode='w')
        save_pkl_dump(f"comet/{split}_e_startend", start_end_embeddings)
        save_pkl_dump(f"comet/{split}_e_cont", continuous_embeddings)
        save_pkl_dump(f"comet/{split}_e_widths", widths)


def gpt3_roberta_embeddings(bert_tokenizer, bert_model, gpt3_inferences_root="gpt3", embedding_mode="ind", max_inferences=5):
    """

    @param bert_tokenizer:
    @type bert_tokenizer:
    @param bert_model:
    @type bert_model:
    @param gpt3_inferences_root:
    @type gpt3_inferences_root:
    @param embedding_mode:
    @type embedding_mode:
    @return:
    @rtype:
    """
    start_end_embeddings = {}
    continuous_embeddings = {}
    widths = {}
    for split in ['train', 'dev']:
        df = pd.read_csv(f'{gpt3_inferences_root}/output_{split}.csv')
        print(f"Number of examples in {split} {df.shape[0]}")
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            key = (row["combined_id"], row["event"])
            all_expansions = row["predictions"]
            inferences = all_expansions.split("After:")
            for i in range(len(inferences)):
                inferences[i] = text_processing(inferences[i])

            before_array = [inf.lstrip()+"." for inf in inferences[0].split(".") if len(inf.split()) > 3] 
            after_array = [inf.lstrip()+"." for inf in inferences[1].split(".") if len(inf.split()) > 3]
            if not before_array:
                before_array = ["."]
            if not after_array:
                after_array = ["."]
            before_array = sorted(before_array[:max_inferences], reverse=False)
            after_array = sorted(after_array[:max_inferences], reverse=False)

            before_condensed = " ".join(before_array).lstrip(". ") + "."
            after_condensed = " ".join(after_array).lstrip(". ") + "."

            # Tokenize and embed before and after
            if embedding_mode == "condensed":
                before_token_ids = [bert_tokenizer.encode(before_condensed)]
                after_token_ids = [bert_tokenizer.encode(after_condensed)]
                max_len = max([len(d) for d in (before_token_ids + after_token_ids)])
                before_embeddings, before_lengths = pad_and_read_bert(before_token_ids, bert_model, max_length=max_len)
                after_embeddings, after_lengths = pad_and_read_bert(after_token_ids, bert_model, max_length=max_len)
            else:
                before_token_ids = [bert_tokenizer.encode(inf) for inf in before_array]
                after_token_ids = [bert_tokenizer.encode(inf) for inf in after_array]
                max_len = max([len(d) for d in (before_token_ids + after_token_ids)])
                before_embeddings, before_lengths = pad_and_read_bert(before_token_ids, bert_model, max_len)
                after_embeddings, after_lengths = pad_and_read_bert(after_token_ids, bert_model, max_len)
                before_embeddings = F.pad(before_embeddings, pad=(0, 0, 0, 0, 0, max_inferences - before_embeddings.shape[0]))
                before_lengths = np.pad(before_lengths, (0, max_inferences - before_lengths.shape[0]), 'constant', constant_values=(0))
                after_embeddings = F.pad(after_embeddings, pad=(0, 0, 0, 0, 0, max_inferences - after_embeddings.shape[0]))
                after_lengths = np.pad(after_lengths, (0, max_inferences - after_lengths.shape[0]), 'constant', constant_values=(0))
            # Stack before and after
            embeddings = torch.cat((before_embeddings, after_embeddings), axis=0)
            lengths = np.concatenate((before_lengths, after_lengths), axis=0)
            print(embeddings.shape, lengths.shape)
            if torch.equal(before_embeddings, after_embeddings):
                print("SAME!")
            starts = embeddings[:, 0, :]
            ends = embeddings[:, -1, :]
            start_ends = torch.hstack((starts, ends))
            start_end_embeddings[key] = start_ends.cpu().detach().numpy()
            # print(start_ends.shape)
            continuous_embeddings[key] = embeddings.cpu().detach().numpy()
            widths[key] = lengths
            torch.cuda.empty_cache()

        # hkl.dump(start_end_embeddings, f"gpt3/{split}_e_startend_ns.hkl", mode='w')
        # hkl.dump(widths, f"gpt3/{split}_e_widths_ns.hkl", mode='w')
        save_pkl_dump(f"gpt3/gpt3_{embedding_mode}_{split}_startend", start_end_embeddings)
        save_pkl_dump(f"gpt3/gpt3_{embedding_mode}_{split}_widths", widths)
        save_pkl_dump(f"gpt3/gpt3_{embedding_mode}_{split}_cont", continuous_embeddings)
        print(f"Done {split}")


# def gpt3_roberta_separate_embeddings(bert_tokenizer, bert_model, gpt3_inferences_root="gpt3"):
#     start_end_embeddings = {}
#     continuous_embeddings = {}
#     widths = {}
#     PADDING = True
#     for split in ['train', 'dev']:
#         df = pd.read_csv(f'gpt3_inferences_root/output_{split}.csv')
#         for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#             key = (row["combined_id"], row["event"])
#             all_expansions = row["predictions"]
#             inferences = all_expansions.split("After:")
#             for i in range(len(inferences)):
#                 inferences[i] = text_processing(inferences[i])

#             before_array = sorted([inf + "." for inf in inferences[0].split(".") if len(inf.split()) > 3])[:5]
#             after_array = sorted([inf + "." for inf in inferences[1].split(".") if len(inf.split()) > 3])[:5]
#             # before_array = before_array + [" "]*(7-len(before_array))
#             # after_array = after_array + [" "]*(7-len(after_array))
#             # print("before, after arrays", len(after_array), len(after_array))
#             final_array = before_array + after_array
#             final_array.sort()
#             token_ids = []
#             for inf in final_array:
#                 tokens = bert_tokenizer.encode(inf)
#                 # print(len(tokens))
#                 token_ids.append(tokens)
#             # print(bert_tokenizer.pad_token_id)
#             # Find all embeddings of the inferences and find start_end_embeddings

#             embds, l = pad_and_read_bert(token_ids, bert_model)
#             embeddings = embds
#             lengths = l
#             if PADDING:
#                 embeddings = F.pad(embds, pad=(0, 0, 0, 0, 0, 10 - embds.shape[0]))
#                 lengths = np.pad(l, (0, 10 - l.shape[0]), 'constant', constant_values=(0))
#             print(embeddings.shape)

#             starts = embeddings[:, 0, :]
#             ends = embeddings[:, -1, :]
#             start_ends = torch.hstack((starts, ends))
#             start_end_embeddings[key] = start_ends.cpu().detach().numpy()
#             # print(start_ends.shape)
#             continuous_embeddings[key] = embeddings.cpu().detach().numpy()
#             # print(embeddings.shape)
#             widths[key] = lengths
#             torch.cuda.empty_cache()
#             # break

#         # # hkl.dump(start_end_embeddings, f"gpt3/{split}_e_startend_ns.hkl", mode='w')
#         # # hkl.dump(widths, f"gpt3/{split}_e_widths_ns.hkl", mode='w')
#         save_pkl_dump(f"gpt3/{split}_e_startend_ind", start_end_embeddings)
#         save_pkl_dump(f"gpt3/{split}_e_widths_ind", widths)
#         save_pkl_dump(f"gpt3/{split}_e_cont_ind", continuous_embeddings)
#         print(f"Done {split}")


def roberta_sentence_embeddings(bert_tokenizer, bert_model):
    start_end_embeddings = {}
    continuous_embeddings = {}
    widths = {}
    for split in ['train', 'val']:
        print("Saved embeddings, saving original sentences")
        original_sentences = dict(
            zip(list(sentences[split]["combined_id"].values), list(sentences[split]["sentence"].values)))
        for key, expansions in tqdm(original_sentences.items()):
            sentence = original_sentences[key]
            token_ids = [bert_tokenizer.encode(sentence)]
            # Find all embeddings of the inferences and find start_end_embeddings
            embeddings, lengths = pad_and_read_bert(token_ids, bert_model)
            starts = embeddings[:, 0, :]
            ends = embeddings[:, -1, :]
            start_ends = torch.hstack((starts, ends))
            start_end_embeddings[key] = start_ends.cpu().detach().numpy()
            continuous_embeddings[key] = embeddings.cpu().detach().numpy()
            widths[key] = lengths
        save_pkl_dump(f"comet/{split}_sent_startend", start_end_embeddings)
        save_pkl_dump(f"comet/{split}_sent_cont", continuous_embeddings)
        save_pkl_dump(f"comet/{split}_sent_widths", widths)


def text_processing(inference):
    inference = inference.replace("After:", "")
    inference = inference.replace("Before:", "")
    inference = inference.replace("After,", "")
    inference = inference.replace("Before,", "")
    inference = inference.replace("\n", "")
    inference = inference.strip()
    return inference


if __name__ == '__main__':
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
    embeddings, lengths = pad_and_read_bert([ids], bert_model)
    print("Sample roberta embeddings size", embeddings.size())

    if commonsense_model == "comet":
        comet_to_roberta_embeddings(bert_tokenizer, bert_model)
    elif commonsense_model == "gpt3":
        # You can embed individually with "ind" and as one single before or after vector with condensed
        gpt3_roberta_embeddings(bert_tokenizer, bert_model, embedding_mode="condensed", max_inferences=5)
    else:
        raise ValueError("commonsense_model should be one of comet or gpt3")

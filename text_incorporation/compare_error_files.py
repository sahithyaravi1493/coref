import pandas as pd
from expansion_embeddings import text_processing
import numpy as np
import ast
import torch
pd.set_option('display.expand_frame_repr', False)
np.set_printoptions(precision=2)

base = 'pairwise_baseline'
new_version = 'pairwise_interspan_2h'


if __name__ == '__main__':
    # Spans and span expansions
    spans = pd.read_csv(f"/ubc/cs/research/nlp/sahiravi/coref/logs/{new_version}/span_examples_ns.csv")
    spans["exps"] = spans["exps"].astype(str)


    # baseline version
    errors1 = pd.read_csv(f'/ubc/cs/research/nlp/sahiravi/coref/logs/{base}/errors.csv')
    print("# of errors in baseline:", errors1.shape[0])
    # print(errors1.columns)

    # new version
    errors2 = pd.read_csv(f'/ubc/cs/research/nlp/sahiravi/coref/logs/{new_version}/errors.csv')
    print("# of errors in newversion:", errors2.shape[0])
    # print(errors2.columns)

    # what is common with baseline
    common = pd.merge(errors1, errors2, how='inner', on=['c1','c2','span1', 'span2'])
    print("# of errors common:",  common.shape[0])
    
    # print(common.head())

    # indicators
    # errors1['indicator'] = errors1['c1'].str.cat(errors1['span1']).cat(errors1['c2']).str.cat(errors1['span2'])
    # errors2['indicator'] = errors2['c1'].str.cat(errors2['span1']).cat(errors2['c2']).str.cat(errors2['span2'])
    # common['indicator'] = common['c1'].str.cat(common['span1']).cat(common['c2']).str.cat(common['span2'])

    # ATTN files:
    attn = pd.read_csv(f'/ubc/cs/research/nlp/sahiravi/coref/logs/{new_version}/attnention.csv')

    F1 = errors2.shape[0]-common.shape[0]
    F2 = errors1.shape[0]-common.shape[0]
    print(f"# errors more than baseline: {F1}")
    print(f"# of errors less than baseline: {F2}")


    # print(common['Unnamed: 0_y'] == errors2.index)
    # print(errors2.head())
    baseline_fails = errors1[~errors1['Unnamed: 0'].isin(common['Unnamed: 0_x'])]
    concat_fails = errors2[~errors2['Unnamed: 0'].isin(common['Unnamed: 0_y'])]

    print("# errors rectified by new version:", baseline_fails.shape[0])
    p = 0
    n = 0
    for index, row in concat_fails.iterrows():
        if row["actual_labels"] == "tensor(1, device='cuda:0', dtype=torch.int32)":
            p += 1

        print("######################### EXAMPLE ##################")
        print(f'{row["actual_labels"]}')
        print(f'{row["sent1"]} \n  {row["span1"]}\n')
        print(f'{row["sent2"]} \n {row["span2"]}\n')

        expansions1 = (spans[(spans["combined_id"] == row["c1"]) & (spans["spans"] == row["span1"])]["exps"].values[0]).split(".")
        expansions2 = (spans[(spans["combined_id"] == row["c2"]) & (spans["spans"] == row["span2"])]["exps"].values[0]).split(".")
        expansions1 = expansions1 + ["NONE"]*(5-len(expansions1))
        expansions2 = expansions2 + ["NONE"]*(5-len(expansions2))

        b1 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["b1"].values[0])
        a1 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["a1"].values[0])
        b2 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["b2"].values[0])
        a2 = ast.literal_eval(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"]) & (attn["c2"] == row["c2"]) & (attn["span2"] == row["span2"])]["a2"].values[0])

        # if b1 and a1 and b2 and a2:
        combined1 = list(zip(expansions1, b1+a1))
        combined2 = list(zip(expansions2, b2+a2))
        combined1 = sorted(combined1, key=lambda l:l[1], reverse=True)
        combined2 = sorted(combined2, key=lambda l:l[1], reverse=True)
        
        print("First span inference:")
        for c in combined1:
            print(c)
        # print(b1 + a1)
        print("Second span inference:")
        # print(expansions2, "\n")
        for c in combined2:
            print(c)
        print("#############################\n")


    print("# of positive error corrections", p)
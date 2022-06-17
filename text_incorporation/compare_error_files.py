import pandas as pd
pd.set_option('max_colwidth', 1000)
# SPANS
spans = pd.read_csv("/ubc/cs/research/nlp/sahiravi/coref/logs/pairwise_interspan/span_examples_ns.csv")

# BASELINE
errors1 = pd.read_csv('/ubc/cs/research/nlp/sahiravi/coref/logs/pairwise_base/errors.csv')
print(errors1.head())
print(errors1.columns)

# CONCAT version
errors2 = pd.read_csv('/ubc/cs/research/nlp/sahiravi/coref/logs/pairwise_interspan/errors.csv')
print(errors2.head())
print(errors2.columns)
# what is common with baseline
common = pd.merge(errors1, errors2, how='inner', on=['c1','c2','span1', 'span2'])
print(common.columns)

# ATTN files:
attn = pd.read_csv('/ubc/cs/research/nlp/sahiravi/coref/logs/pairwise_interspan/attnention.csv')

F1 = errors2.shape[0]-common.shape[0]
F2 = errors1.shape[0]-common.shape[0]
print(f"Concat has {F1} errors more than baseline")
print(f"Baseline has {F2} more than concat")
baseline_fails = errors1[~errors1.index.isin(common.index)]
concat_fails = errors2[~errors2.index.isin(common.index)]

print("cases where knowledge helps", baseline_fails.shape)
p = 0
n = 0
for index, row in  baseline_fails.iterrows():
    if row["actual_labels"] == 0:
        n += 1
    print(f'{row["actual_labels"]}')
    print(f'{row["sent1"]} ###### {row["span1"]}\n')
    print(f'{row["sent2"]} ###### {row["span2"]}\n')
   
    print(spans[(spans["combined_id"] == row["c1"]) & (spans["spans"] == row["span1"])]["exps"].values[0], "\n")
    print(spans[(spans["combined_id"] == row["c2"]) & (spans["spans"] == row["span2"])]["exps"].values[0], "\n")
    print(attn[(attn["c1"] == row["c1"]) & (attn["span1"] == row["span1"])]["b1"].head(1), "\n")
    print(attn[(attn["c1"] == row["c2"]) & (attn["span2"] == row["span2"])]["b2"].head(1), "\n")
    print("#############################\n")
print(n, baseline_fails.shape[0]-n)
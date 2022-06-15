import pandas as pd
# SPANS
spans = pd.read_csv("/ubc/cs/research/nlp/sahiravi/coref/logs/pairwise_weighted_concat/span_examples_ns.csv")

# BASELINE
errors1 = pd.read_csv('/ubc/cs/research/nlp/sahiravi/coref/logs/pairwise_weighted/errors.csv')
print(errors1.head())
print(errors1.columns)

# CONCAT version
errors2 = pd.read_csv('/ubc/cs/research/nlp/sahiravi/coref/logs/pairwise_weighted_concat/errors.csv')
print(errors2.head())
print(errors2.columns)
# what is common with baseline
common = pd.merge(errors1, errors2, how='inner', on=['c1','c2','span1', 'span2'])
print(common.columns)

F1 = errors2.shape[0]-common.shape[0]
F2 = errors1.shape[0]-common.shape[0]
print(f"Concat has {F1} errors more than baseline")
print(f"Baseline has {F2} more than concat")
baseline_fails = errors1[~errors1.index.isin(common.index)]
concat_fails = errors2[~errors2.index.isin(common.index)]

print("cases where knowledge helps", baseline_fails.shape)
c = 0
for index, row in  baseline_fails.iterrows():
    c += 1
    print(f'{row["actual_labels"]}')
    print(f'{row["sent1"]} ###### {row["span1"]}\n')
    print(f'{row["sent2"]} ###### {row["span2"]}\n')
   
    print(spans[(spans["combined_id"] == row["c1"]) & (spans["spans"] == row["span1"])]["exps"].values[0], "\n")
    print(spans[(spans["combined_id"] == row["c2"]) & (spans["spans"] == row["span2"])]["exps"].values[0], "\n")
    print("#############################\n")

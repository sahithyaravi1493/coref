# 1 ENV: Activate your env
source coref/bin/activate

# 2 CONFIG: configure seed and paths
seed="120" # make sure you change logs, models to _seed in config files
configseed=""  # optional if config is renamed
model="interspan"
gpu=1

# 3 TRAIN: Run pairwise scorer
CUDA_VISIBLE_DEVICES=$gpu  python -u train_pairwise_scorer.py --config "configs/${model}/config_pairwise${configseed}.json"

# 4 TUNE CLUSTER: Find clustering threshold on dev set
CUDA_VISIBLE_DEVICES=$gpu nohup python -u tuned_threshold.py --config "configs/${model}/config_clustering${configseed}.json"

# 5 BEST MODEL: Find best model on dev set, take the last printed line as it contains the threshold, modelnum
str=$(python run_scorer.py "models/${model}_${seed}" events)
s="${str##*$'\n'}"
echo $s

# 6 PREDICT: Run predict.py with the best model from previous step
# This is automated but you can manually configure congig_clustering....json file
str=$(CUDA_VISIBLE_DEVICES=$gpu python -u predict.py --config "configs/${model}/config_clustering${configseed}.json" --dev_best_name "${s}")
s="${str##*$'\n'}"

# 7 EVAL: Compute test score
python -u coval/scorer.py data/ecb/gold_singletons/test_events_topic_level.conll  "models/${model}_${seed}/${s}"
mv "inter.out" "logs/${model}_${seed}"
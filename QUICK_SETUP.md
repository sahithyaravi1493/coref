
## Getting started
Clone the knowledge addition branch:
`git clone -b knowledge_addition https://github.com/sahithyaravi1493/coref.git`

* Install python3 requirements `pip install -r requirements.txt` 


### Extract mentions and raw text from ECB+ 

Run the following script in order to extract the data from ECB+ dataset
 and build the gold conll files. 
The ECB+ corpus can be downloaded [here](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).

```
python get_ecb_data.py --data_path path_to_data
```

### Commonsense knowledge
place the commonsense inferences for gpt3 under gpt3/ folder
coref/gpt3/output_dev.csv
coref/gpt3/output_train.csv
coref/gpt3/output_test.csv
Next run python text_incorporation/expansion_embeddings.py
It should save the embeddings for all the splits as a pkl file.

## Training Instructions
The core of our model is the pairwise scorer between two spans, 
which indicates how likely two spans belong to the same cluster.


#### Training method

For commonsense, we will use only the gold mentions - hence we will set the configuration files accordingly
- include_text: setting this to false runs the baseline, setting it to true adds the knowledge embeddings
- fusion: can be 'intraspan', 'interpsan' or 'inter_intra' for different attention options
- inferences_path, text_embeddings_path - indidcates where the cs inferences, embeddings are stored.


In order to choose the training method, you need to set the value of the `training_method` in 
the `config_pairwise.json` to `pipeline` for the gold mentions. There is not need to use continual training as the span scoring is not needed for gold mentions.
 
#### What are the labels ?

In ECB+, the entity and event coreference clusters are annotated separately, 
making it possible to train a model only on event or entity coreference. 
Therefore, our model also allows to be trained on events, entity, or both.
You need to set the value of the `mention_type` in 
the ``config_pairwise.json`` (and `config_span_scorer.json`) 
to `events`, `entities` or `mixed` (corresponding to ALL in the paper).


#### Running the model
 

For the pairwise scorer, run the following script
```
python train_pairwise_scorer.py
```
It will save 10-15 different models for the span, pairwise, fusion models/

Some important parameters in `config_pairwise.json`:
* `max_mention_span`
* `top_k`: pruning coefficient
* `training_method`: (pipeline, continue, e2e)
* `subtopic`: (true, false) whether to train at the topic or subtopic level (ECB+ notions).


#### Tuning threshold for agglomerative clustering

In order to find the best model and the best threshold for the agglomerative clustering, 
you need to do an hyperparameter search on the 10 models + several values for threshold, 
evaluated on the dev set. To do that, please set the `config_clustering.json` (`split`: `dev`) 
and run the two following scripts:

You need to edit the path of the model in  `config_clustering.json` to the appropriate model from previous step.
You also need to set `include_text` as True and fusion to the right method for using commonsense, and 
```
python tuned_threshold.py 
python run_scorer.py [path_of_directory_of_conll_files] [mention_type]
```


## Prediction

Given the trained pairwise scorer, the best `model_num` and the `threshold` 
from the above training and tuning, set the `config_clustering.json` (`split`: `test`)
and run the following script. 

```
python predict.py --config configs/config_clustering
```

(`model_path` corresponds to the directory in which you've stored the trained models)

An important configuration in the `config_clustering` is the `topic_level`. 
If you set `false` , you need to provide the path to the predicted topics in `predicted_topics_path` 
to produce conll files at the corpus level. 

## Evaluation

The output of the `predict.py` script is a file in the standard conll format. 
For evaluation run
`python coval/scorer.py  /ubc/cs/research/nlp/sahiravi/coref/data/ecb/gold_singletons/dev_events_topic_level.conll ` [your best conll model]


## Notes
* If you chose to train the pairwise with the end-to-end method, you don't need to provide a `span_repr_path` or a `span_scorer_path` in the
`config_pairwise.json`.  

* If you use this model with gold mentions, the span scorer is not relevant, you should ignore the training method.

* If you're interested in a newer but heavier model, check out our [cross-encoder model](https://github.com/ariecattan/cross_encoder/)



## Team 

* [Arie Cattan](https://ariecattan.github.io/)
* Alon Eirew
* [Gabriel Stanovsky](https://gabrielstanovsky.github.io/)
* [Mandar Joshi](https://homes.cs.washington.edu/~mandar90/)
* [Ido Dagan](https://u.cs.biu.ac.il/~dagan/) 



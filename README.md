# TAFSSL
Task-Adaptive Feature Sub-Space Learning for few-shot classification

https://arxiv.org/abs/2003.06670

# Code for the experiments 
## (according to tables and figures in the paper):

## Table 1: Transductive setting
python exp\_table.py

## Table 2: Semi supervised setting
python exp\_semi.py

## Figure 2: Number of queries in transductive FSL setting
python exp\_num\_query.py

## Figure 3: The affect of the unlabeled data noise on the performance
python exp\_noise\_semi.py

## Figure 4: ICA dimension vs accuracy
python exp\_projection\_dim.py

## Figure 5: Unbalanced
python exp\_unbalanced.py

# To re-create the feature files:
    1. Download miniImageNet / tieredImageNet
    2. Generate splits:
       python src/utils/tieredImagenet.py --data path-to-tiered --split split/tiered/
    3. Pre-train a model and store the features:
       python ./src/train.py -c $<$path to config file$>$
       python ./src/train.py -c $<$path to config file$>$ --save-features --enlarge

# License
Copyright 2019 IBM Corp. This repository is released under the Apachi-2.0 license (see the LICENSE file for details)

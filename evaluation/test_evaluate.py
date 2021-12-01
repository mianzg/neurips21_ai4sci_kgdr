# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 20:26:41 2021

@author: Superhhu
"""



from pykeen.hpo import hpo_pipeline
from pykeen.datasets import PathDataset
from pykeen.evaluation import RankBasedEvaluator
from pykeen.datasets.analysis  import get_entity_count_df



import torch


hetionet = PathDataset(training_path="hetionet_training_new.tsv",
                       validation_path="hetionet_validation_new.tsv",
                       testing_path="hetionet_testing_new.tsv"
)
train_treat = hetionet.training.new_with_restriction(relations={'CtD'})
validation_treat = hetionet.validation.new_with_restriction(relations={'CtD'})
testing_treat = hetionet.testing.new_with_restriction(relations={'CtD'})


testing_treat_actual=testing_treat.mapped_triples
testing_treat_actual=testing_treat_actual.numpy()

import pickle
with open('./hetionet/entity_mapping.pkl', 'rb') as fp:
    entity_mapping = pickle.load(fp)
    
model  = torch.load('rotate.pkl',map_location ='cpu')
model.device='cpu'
model.triples_factory=testing_treat

all_predictions=[]
for i in testing_treat_actual:
    drug_name=entity_mapping[i[0]]
    predictions=model.get_tail_prediction_df(drug_name,"CtD",triples_factory=testing_treat)
    all_predictions.append(predictions)
    

with open("total_result_ctd_rotate.pkl", 'wb') as handle:
    pickle.dump(all_predictions, handle)



"""analysis of class"""
top10=[]
top10_class=[]
for i in total_result:
    temp=i.iloc[0:10]['tail_label']
    for j in temp:
        top10_class.append(j.split("::")[0])
    top10.extend(i.iloc[0:10]['tail_label'])

"""% of disease at 10"""
np.unique(top10_class,return_counts=True)
top1=[]
for i in total_result:
    top1.append(i.iloc[0]['tail_label'])
    
    
"""unique entity at 1"""
len(np.unique(top1))

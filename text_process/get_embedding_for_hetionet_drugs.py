# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:40:07 2021

@author: Superhhu
"""

from transformers import AutoTokenizer, AutoModel
  
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")


import json
import glob
from pykeen.datasets import Hetionet
import pandas as pd
from pykeen.datasets.analysis  import *
import numpy as np 
from tqdm import tqdm
test1=Hetionet(random_state=42)

entity_name=get_entity_count_df(test1)

hetionet_druglist=[]
for index,i in entity_name.iterrows():

    hetionet_druglist.append(i[2])
        


text_entity_list=glob.glob(r'./hetionet/text_process/*.json')
all_textentity={}
count=[]
for i in text_entity_list:
    with open(i) as f:
        data = json.load(f)
        all_textentity = {**all_textentity, **data}



all_textentity=[(k,v) for k,v in all_textentity.items()]


result=all_textentity
all_text=[]
all_outputs=[]
for i in tqdm(result):
    idx,text=i
    if text:
    
        encodings = tokenizer(text, max_length=512,truncation=True, padding=True,return_tensors="pt")
        outputs=model(**encodings)[1]
        outputs=np.squeeze(outputs.cpu().detach().numpy())
        all_outputs.append((idx,outputs))



import pickle

with open('drug_text_new.pkl', 'wb') as f:
    pickle.dump(all_outputs, f)

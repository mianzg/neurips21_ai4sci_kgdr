# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 21:46:51 2021

@author: Superhhu
"""

from pykeen.pipeline import pipeline
from pykeen.datasets import PathDataset,EagerDataset
import pickle
with open('./hetionet/drug_text_new.pkl', 'rb') as f:
    textliteral= pickle.load(f)
import torch

from pykeen.triples import TriplesTextualLiteralsFactory,TriplesFactory


hetionet = PathDataset(training_path="./hetionet/hetionet_training_new.tsv",
                    validation_path="./hetionet/hetionet_validation_new.tsv",
                    testing_path="./hetionet/hetionet_testing_new.tsv"
)      

training = TriplesTextualLiteralsFactory(triples=hetionet.training.triples, textual_triples=textliteral,
                                         entity_to_id=hetionet.entity_to_id,
    relation_to_id=hetionet.relation_to_id,
)

validation = TriplesTextualLiteralsFactory(triples=hetionet.validation.triples, textual_triples=textliteral,
                                           entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,

)
testing = TriplesTextualLiteralsFactory(triples=hetionet.testing.triples, textual_triples=textliteral,
                                        entity_to_id=training.entity_to_id,
    relation_to_id=training.relation_to_id,

)


hetionet = EagerDataset(training=training,
                    validation=validation,
                    testing=testing
)    


pipeline(    
        random_seed=42,
    evaluator= "rankbased",
    loss= "marginranking",
    loss_kwargs= {
      "margin": 1
    },
    model= "DistMultText",
    model_kwargs= {
      "embedding_dim": 214,
    },
    negative_sampler= "basic",
    negative_sampler_kwargs= {
          #  "filtered":True,
      "num_negs_per_pos": 61,
    },
    optimizer= "Adagrad",
    optimizer_kwargs= {
      "lr": 0.030
    },
    testing= testing,
    training= training,
    training_kwargs= {
      "batch_size": 512,
      "num_epochs": 400,
      "checkpoint_name": 'checkpoint_distmult_text_V0.pt',
      "checkpoint_frequency": 1,
      "checkpoint_directory": "./checkpoint_distmult_text_V0",
    },
    result_tracker="wandb",
    result_tracker_kwargs={
        "project":"CSNLP",
    },
    training_loop= "slcwa",
    stopper = "early",
    stopper_kwargs = {
        "frequency": 1,
        "patience": 2000,
        "relative_delta": 0.001,
        "metric": "mean_reciprocal_rank"
    },
    validation= validation).save_model("./checkpoint_distmult_text_V0/distmult_text.pkl")

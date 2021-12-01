from pykeen.pipeline import pipeline
from pykeen.datasets import PathDataset
from pykeen.hpo import hpo_pipeline

import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)


hetionet = PathDataset(training_path="./hetionet_new/hetionet_training_new.tsv",
                    validation_path="./hetionet_new/hetionet_validation_new.tsv",
                    testing_path="./hetionet_new/hetionet_testing_new.tsv"
)      


hpo_result = hpo_pipeline(
    n_trials=10,
    study_name="simple",
     storage= "sqlite:///simple.db",
     load_if_exists= "True",
    training=hetionet.training,
    testing=hetionet.testing,
    validation=hetionet.validation,
    model='simple',
    model_kwargs_ranges=dict(
        embedding_dim=dict(type=int, low=200, high=500, q=20)
    ),
    optimizer= "Adagrad",
    optimizer_kwargs_ranges=dict(
        lr=dict(type='float', low=0.001, high=0.1, scale="log")
    ),
    training_loop='slcwa',
    training_kwargs=dict(num_epochs=50,batch_size=512),
    stopper = "early",
    stopper_kwargs = {
        "frequency": 5,
        "patience": 20,
        "relative_delta": 0.002,
        "metric": "mean_reciprocal_rank"
    },
)
hpo_result.save_to_directory("./hpo_simple")


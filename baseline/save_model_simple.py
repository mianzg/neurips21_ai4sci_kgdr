from pykeen.pipeline import pipeline
from pykeen.datasets import PathDataset

import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)


hetionet = PathDataset(training_path="./hetionet/hetionet_training_new.tsv",
                    validation_path="./hetionet/hetionet_validation_new.tsv",
                    testing_path="./hetionet/hetionet_testing_new.tsv"
)      


pipeline(   
    random_seed= 42, 
    evaluator= "rankbased",
    loss= "softplus",
    model= "simple",
    model_kwargs= {
      "embedding_dim": 380,
    },
    negative_sampler= "basic",
    negative_sampler_kwargs= {
      "num_negs_per_pos": 91,
    },
    optimizer= "Adagrad",
    optimizer_kwargs= {
      "lr": 0.012712788825076992
    },
    regularizer= "powersum",
    regularizer_kwargs= {
      "weight": 0.10127031352995612
    },
    testing= hetionet.testing,
    training= hetionet.training,
    training_kwargs= {
      "batch_size": 512,
      "num_epochs": 500,
      "checkpoint_name": 'checkpoint_simple.pt',
      "checkpoint_frequency": 240,
      "checkpoint_directory": "./checkpoint_dir_simple",
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
    validation= hetionet.validation).save_model("./simple.pkl")



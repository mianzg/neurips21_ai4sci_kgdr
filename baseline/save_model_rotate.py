from pykeen.pipeline import pipeline
from pykeen.datasets import PathDataset

hetionet = PathDataset(training_path="./hetionet/hetionet_training_new.tsv",
                    validation_path="./hetionet/hetionet_validation_new.tsv",
                    testing_path="./hetionet/hetionet_testing_new.tsv"
)     

import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)


pipeline(    
    random_seed= 42,
    evaluator= "rankbased",
    loss= "marginranking",
    loss_kwargs= {
      "margin": 1
    },
    model= "rotate",
    model_kwargs= {
      "embedding_dim": 483,
    },
    negative_sampler= "basic",
    negative_sampler_kwargs= {
      "num_negs_per_pos": 31
    },
    optimizer= "Adagrad",
    optimizer_kwargs= {
      "lr":  0.028
    },
    regularizer= "no",
    testing= hetionet.testing,
    training= hetionet.training,
    training_kwargs= {
      "batch_size": 512,
      "num_epochs": 840,
      "checkpoint_name": 'checkpoint_rotate.pt',
      "checkpoint_frequency": 240,
      "checkpoint_directory": "./checkpoint_dir_rotate",
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
    validation= hetionet.validation).save_model("./rotate.pkl")


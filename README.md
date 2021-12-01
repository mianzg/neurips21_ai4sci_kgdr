# Drug Repositioning via Text Augmented KG Embeddings
Code repository to **NeurIPS 2021 AI for Science Workshop**

## Repository Setup
1. Install required libraries with
```
pip install pykeen==1.5.0 wandb
```
Refer to [link](https://pykeen.readthedocs.io/en/stable/tutorial/trackers/using_wandb.html) for more instructions on WANDB result tracker. 
You will be able to see various metrics during training and the evaluation after training. 

2. Download files into the corresponding directory under root
- [Data and Trained Models](https://polybox.ethz.ch/index.php/s/5gT0fMMUqS6sMnA) 


## Baseline Training
1. Train and save trained baseline embedding model X (transe, distmult, proje, rotate, simple, tucker)
```
python baseline/save_model_X.py
```

2. Example hyperparameter optimization 
```
python baseline/hpo_search.py
```

## Text Augmented Model Training
1. (optional) Scrape text from sources 

The needed json files of scraped text are already in the directory. If you are interested in doing from scratch, first download the [hetionet json dataset](https://github.com/hetio/hetionet#license)
```
python text_scrape.py -n X
```
where X must be one of the available entity type('Anatomy', 'Biological Process', 'Cellular Component', 'Compound', 'Disease', 'Gene'(local), 'Molecular Function', 'Pathway', 'Pharmacologic Class')

2. Text Embedding 
We provide the texual embedding generated with BioBert V1.1 in the aforementioned polybox folder. If you are interested in generate this texual embedding yourself,
please refer to 

```text_process/get_embedding_for_hetionet_drugs.py```


You will need to install Hugging Face Transformers library. 

3. Find your pykeen library installation path and replace corresponding files with the ones in ```/pykeen-extension/```
This step is to include the texual interaction etc. functionality. 

4. Train Text Augmented KG 
Run the model you desire to train by  
```python model_with_text/X.py``` 
The file name should be self-explanatory.

5. To calculate % of Disease @10 and Unique Entities @1:
please refer to 

```evaluation/test_evaluate.py```

# COMAL-small  

## Introduction  

This is a trivial reproduction of the paper [CoMAL: Contrastive Active Learning for Multi-Label Text
Classification](https://markdown.com.cn) on [PubMed dataset](https://huggingface.co/datasets/owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH).   

This repository is served as part of our course project and is NOT intended for use in formal research.  

There are already [official implementation](https://github.com/chengzju/CoMAL) for this paper. Please take that as first choice.  


## Results  
If you just wants to check the result of the course projects, you can ignore the rest part of this readme, prepare an environment with tensorboard installed and find our related logs at `result_logs/`.  
For example, run:  
```bash
tensorboard --logdir=result_logs/official_code_on_rcv1 --port=2024
```


## Environment setup  
```bash
conda env create -f requirements.yaml
```
This code only has some very basic dependencies, however you may still need to adjust the version of some packages if this setting is incompatible with your machine.  

If you are using pytorch 1.x, please comment all the lines with `torch.complie`. If you are using pytorch 2.x you can do that as well because it seems not making the model faster.   

## Data preperation  

* Download the [PubMed dataset](https://huggingface.co/datasets/owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH), name the csv file `pm.csv` and put it under `data/PubMed/pm.csv`.  

* The csv file is already a preprocessed version and our code for dataset can handle some exceptions, so there is no need for extra data preprocessing.  

## Experiments  

### Train a simple multi-label text classifier  

To train a simple multi-label text classifier without contrastive learning module and active learning sampling strategy:   
```
python main_trivial.py
```

### Train a multi-label text classifier chained with contrastive learning module  

To validate the effect of chaining the contrastive module and the backbone together, you may want to run this experiments and compare it with the simple classifier:  
```bash
python main_contrastive.py
```

### Train a full model with random sampling strategy  

Please first set the boolean variable `fake_active=True` in `main_active.py` then run it:  
```bash
python main_active.py
```

### Train a full model with proposed active learning sampling strategy  

Please first set the boolean variable `fake_active=False` in `main_active.py` then run it:  
```bash
python main_active.py
```

---  

### Hope you enjoy though it's unlikely you may like this repo :).  
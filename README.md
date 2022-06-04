<div align="center">    
 
# Deep Traffic Pytorch

<!--
Conference
-->
</div>
 
## Description   
Pytorch version from https://github.com/echowei/DeepTraffic <br>
You can simplify analyze all "mnist" dataset from DeepTraffic using pytorch and pytorch lightening

## How to run

Run on Python 3.8 <br>
First, install dependencies

```bash
# clone project
git clone https://github.com/muchlisinadi/DeepTraffic-pytorch

# install project
cd DeepTraffic-pytorch
conda env update -n deeptraffic --file environment.yaml
```

Next, navigate to any notebook file and run it.

## Imports

This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from project.datasets.mnist import mnist
from project.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()

# data
train, val, test = mnist()

# train
trainer = Trainer()
trainer.fit(model, train, val)

# test using the best model!
trainer.test(test_dataloaders=test)
```

### Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
```

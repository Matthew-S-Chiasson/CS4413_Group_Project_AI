# CS4413_Group_Project_AI

## Imact of DP and HE on Model preformance and effitency.

# Purpose

The idea behind this project is to give a visual representation of the trade off between accurcy and privacy.

# Method

# Dataset

CUB-17 is a truncation of the Caltech-UCSD Birds-200-2011 (CUB-200-2011) dataset.
We drop all but the first 17 classes from the dataset to imporve training time for our project.

# Victum Model

  We train 3 different victum models. Each model uses resnet18.
  
  ## Modified Traier:
    Uses a modified architecture of resnet18. We add Linerar regression, a dropout layer and, ... to imporve model accuracy.
    Resnet18 is imported wihtout its learned paramiters.
    Achives accuracy of ~80% on CUB17 dataset
  
  ## Model Trainer:

    Uses an unmodified resnet18 model. No extra fully corrected layers.
    Resnet18 is imported wihtout its learned paramiters.
    Achives accuracy of ~65% on CUB17 dataset
    
  ## Pre-T trainer:

    Uses an unmodified resnet18 model. No extra fully corrected layers.
    Resnet18 is imported wiht its learned paramiters.
    Achives accuracy of ~91% on CUB17 dataset

Each victum model is trained and saved in the format: resnet18_ *dataset used* _ *trainer* _ *Fedarated Learning?* _ *used DP?* _ *Used HE?* _ *accuracy achived* .pth, to be later used in attack simulation.

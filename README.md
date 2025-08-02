# What?

This project applies deep learning to music to uncover genre-defining characteristics and improve genre classification accuracy. My first objective is to build a classifier that can predict the genre of a song based on its audio features, using transformer-based models well-suited for sequential data like music. Beyond classification accuracy, I aim to understand why the model makes its predictions by analyzing the attention vectors. This will allow me to identify which parts of a song are most influential in determining its genre, offering deeper insight into the acoustic patterns that distinguish genres. Ultimately, my goal is not only to develop a high-performing classifier but also to advance interpretability by revealing the specific audio segments that contribute most to genre identification.

# How?

I began by building a genre classifier using a transformer model. I began with a simple model with 2 layers and 4 attention heads, then quickly improved to a 4 layer and 4 head per layer model. I tokenized our audio files using mel spectrograms, a time-frequency representation of audio that show how the frequency content evolves over time; through these we managed to compress each 30 second audio into a (1290,128) tensor, which I then used to train and test our model. I managed to reach a 95% accuracy, precision, and recall on the unseen data. Finally, I extracted the attention vectors from the model to analyze which characteristics are particularly important in predicting what genre an audio is.

# How to run the code

The first thing to focus on is importing the data. I imported the data from Kaggle, using the command:
import kagglehub
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
 This command stores the data at location: path. Note that the following code might have to be modified to accommodate the user's given path. We then moved the data to the present working directory using the command: mv path/* .

The remainder of the code is run simply by going through the python notebook sequentially. 

Note that you will need to have access to a GPU to run this code. 
Note that some of the graphs may vary as the model's final weights depend on the random initialization of the weights. 

# Required dependencies
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import librosa.display

import librosa
import IPython.display as ipd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from joblib import Parallel, delayed


# Describe any pretrained models or datasets used
The dataset used in GTZAN, which contains 1000 audio clips (10 genres, 100 clips each), each 30 seconds long, at 22050Hz, and is the standard benchmark for music genre classification.
I used wav2vec, a pretrained audio transformer designed by Facebook to embed the audio files. However, we quickly veered away from this and used mel spectrograms instead.


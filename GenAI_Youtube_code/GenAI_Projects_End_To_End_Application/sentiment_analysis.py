# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 17:18:20 2024

@author: akash
"""

#!pip -q install accelerate -U
#!pip -q install transformers[torch]
#!pip -q install datasets
#Restart after installing


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from transformers import pipeline




#sentiment analyis model-2
sentiment_model2 = pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest")

data="over heating issue don't buy this product but camera was good"
print(sentiment_model2(data))